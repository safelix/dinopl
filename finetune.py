import argparse
import copy
import os
from typing import Dict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import prune
from torchmetrics import Accuracy
from tqdm import tqdm

from configuration import Configuration
import dinopl.utils as U
import wandb
from dinopl import DINO, DINOHead, DINOModel, probing
from losslandscape import load_model, load_config
from models import Encoder
import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from dinopl.scheduling import Schedule, Scheduler


def get_prune_params(module):
    prune_params = []
    for m in module.modules():
        if getattr(m, 'weight', None) is not None:
            prune_params.append((m, 'weight'))
        if getattr(m, 'bias', None) is not None:
            prune_params.append((m, 'bias'))
    return prune_params

def get_dataloader(dataset:str, augmentations:bool, batch_size:int, num_workers:int, pin_memory:bool):
    if dataset == 'mnist':
        DSet = datasets.MNIST
    if dataset == 'cifar10':
        DSet = datasets.CIFAR10
    if dataset == 'cifar100':
        DSet = datasets.CIFAR100

    transform = transforms.Compose([
                    transforms.Lambda(lambda img: img.convert('RGB')), 
                    transforms.ToTensor(),
                    transforms.Normalize(DSet.mean, DSet.std)])

    if augmentations:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(DSet.img_size, 4),
            transform,
        ])
    
    train_ds = DSet(root=os.environ['DINO_DATA'], train=True, transform=transform, download=True)
    valid_ds = DSet(root=os.environ['DINO_DATA'], train=False, transform=transform, download=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, valid_dl

def get_optimizer(args:dict, params) -> torch.optim.Optimizer:
    # Learning rate and weight decay are set by the scheduler
    # Raise error since lr wouldn't be set
    if args['opt'].lower() == 'sgd' and args['opt_lr'] is None: 
        raise ValueError('SGD requires a learning rate to be set.')

    if args['opt'].lower() == 'sgd':
        return torch.optim.SGD(params, lr=float('inf'), momentum=0.9) 
    if args['opt'].lower() == 'adam':
        return torch.optim.Adam(params)
    if args['opt'].lower() == 'adamw':
        return torch.optim.AdamW(params)

def get_scheduler(args:dict, optimizer:torch.optim.Optimizer) -> Schedule:
    scheduler = Scheduler()
    if isinstance(args['opt_lr'], Schedule):
        scheduler.add(optimizer.param_groups[0], 'lr', args['opt_lr'])
    if isinstance(args['opt_wd'], Schedule):
        scheduler.add(optimizer.param_groups[0], 'weight_decay', args['opt_wd'])

    return scheduler

@torch.no_grad()
def main(args:dict):
    wandb_run:wandb.wandb_sdk.wandb_run.Run = wandb.run # for pylint
    wandb_run.define_metric('train/acc', summary='max')
    wandb_run.define_metric('valid/acc', summary='max')
    device = torch.device('cpu' if args['force_cpu'] else U.pick_single_gpu())

    # Load Data and model
    train_dl, valid_dl = get_dataloader(args['dataset'], args['augmentations'], args['batch_size'], args['num_workers'], not args['force_cpu'])
    args['n_classes'] = train_dl.dataset.ds_classes
    model:Encoder = load_model(args['ckpt']).enc.to(device)
    if isinstance(model, DINO):
        raise ValueError('Checkpoint needs to be either :teacher or :student')

    # Prune encoder
    prune.global_unstructured(get_prune_params(model), prune.L1Unstructured, amount=args['prune_ratio'])
    
    # Prepare Classifier
    lin = probing.LinearAnalysis(args['pre_epochs_clf'])
    lin.prepare(model.embed_dim, train_dl.dataset.ds_classes, device=device)
    #lin.clf = type(model)(num_classes=args['n_classes']).classifier # get new classifier from encoder type

    lin.train(probing.load_data(model, train_dl, device))
    acc = lin.valid(probing.load_data(model, valid_dl, device))
    wandb_run.log({'trainer/epoch':-1, 'trainer/step': -1, 'valid/acc':acc})

    model.classifier = copy.deepcopy(lin.clf) # add trained classifier to encoder
    lin.cleanup()

    # Train Loop
    step = -1
    model.requires_grad_(True)
    optimizer = get_optimizer(args, model.parameters())
    scheduler = get_scheduler(args, optimizer).prep(args['n_epochs'], len(train_dl))
    train_acc, valid_acc = Accuracy().to(device), Accuracy().to(device)
    for epoch in range(args['n_epochs']):
        
        # Training Epoch
        train_acc.reset(),
        progress_bar = tqdm(train_dl, desc=f'Train Epoch {epoch}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            step += 1
            scheduler.step(step)
            optimizer.zero_grad(True)
            with torch.enable_grad():
                predictions = model(inputs)
                loss = F.cross_entropy(predictions, targets)
            loss.backward()
            optimizer.step()

            # process logs
            acc = train_acc(predictions, targets)
            progress_bar.set_postfix({'loss':loss.item(), 'acc':acc.item()})
            wandb_run.log({'trainer/step': step, 'train/acc':acc, 'train/loss':loss,
                            'hparams/lr':optimizer.param_groups[0]['lr'],
                            'hparams/wd':optimizer.param_groups[0]['weight_decay']})

        # Validation Epoch
        valid_acc.reset()
        valid_loss, numel = 0, 0
        progress_bar = tqdm(valid_dl, desc=f'Valid Epoch {epoch}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model(inputs)
            loss = F.cross_entropy(predictions, targets)

            # process logs
            numel += len(targets)
            valid_loss += loss * len(targets)
            valid_acc.update(predictions, targets)
            progress_bar.set_postfix({'loss':loss.item(), 'acc':valid_acc.compute().item()})
        valid_loss = valid_loss / numel


        # Log Epoch
        wandb_run.log({'trainer/epoch':epoch, 'trainer/step': step,
                        'valid/loss':valid_loss, 'valid/acc':valid_acc.compute()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to finetune.')
    parser.add_argument('--prune_ratio', type=float, default=0,
                        help='Ratio of smallest weights to prune.')
    
    # Data Arguments
    parser.add_argument('--dataset', choices={'mnist', 'cifar10'}, default='cifar10')
    parser.add_argument('--augmentations', type=U.bool_parser, default=False)
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for the data loader.')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of parallel threads for data loading.')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=50, 
                    help='Number of epochs to train for.')
    parser.add_argument('--pre_epochs_clf', type=int, default=0,
                        help='Number of epochs to pretrain classifier for.')
    parser.add_argument('--opt', type=str, choices={'adamw', 'adam', 'sgd'}, default='adamw', 
                        help='Optimizer to use for training (float or Schedule)')                   
    parser.add_argument('--opt_lr', type=Schedule.parse, default=None, 
                        help='Learning rate for optimizer (float or Schedule).')
    parser.add_argument('--opt_wd', type=Schedule.parse, default=None, 
                        help='Weight decay for optimizer.')

    parser.add_argument('--force_cpu', action='store_true')
    args = vars(parser.parse_args())

    if args['ckpt'] is None:
        raise ValueError('Please specify checkpoint.')

    # if path is not relative, make relative for saving to wandb
    if not args['ckpt'].startswith('DINO'): 
        args['ckpt'] = os.path.relpath(args['ckpt'], os.environ['DINO_RESULTS']) 

    # init wandb
    args['dino_config'] = vars(load_config(os.path.join(os.environ['DINO_RESULTS'], args['ckpt']))) # add dino_config
    wandb.init(project='DINO_finetune', dir=os.environ['DINO_RESULTS'], config=args)
    args['ckpt'] = os.path.join(os.environ['DINO_RESULTS'], args['ckpt']) # make absolute

    main(args)
