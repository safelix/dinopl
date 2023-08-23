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

from configuration import Configuration, load_model, load_config
import dinopl.utils as U
import wandb
from dinopl import DINO, DINOHead, DINOModel, probing
from models import Encoder
import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from dinopl.scheduling import Schedule, Scheduler


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

def get_scheduler(args:dict, optimizer:torch.optim.Optimizer) -> Scheduler:
    scheduler = Scheduler()
    if isinstance(args['opt_lr'], Schedule):
        scheduler.add(optimizer.param_groups[0], 'lr', args['opt_lr'])
    if isinstance(args['opt_wd'], Schedule):
        scheduler.add(optimizer.param_groups[0], 'weight_decay', args['opt_wd'])

    return scheduler

@torch.no_grad()
def main():
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
    
    # Prepare Classifier
    lin = probing.LinearAnalysis(args['pre_epochs_clf'])
    generator = None if args['seed'] is None else  torch.Generator(device=device).manual_seed(args['seed']) 
    lin.prepare(model.embed_dim, train_dl.dataset.ds_classes, device=device, generator=generator)

    if args['pre_epochs_clf'] > 0:
        lin.train(probing.load_data(model, train_dl, device))
    acc = lin.valid(probing.load_data(model, valid_dl, device))
    wandb_run.log({'trainer/epoch':-1, 'trainer/step': -1, 'valid/acc':acc})

    model.fc = copy.deepcopy(lin.clf) # add trained classifier to encoder
    del lin


    # Prepare pruning: get parameter names and their parent modules for pruning
    p_names = [n for m in model.modules() for n, p in m.named_parameters(recurse=False)]
    p_modules = [m for m in model.modules() for p in m.parameters(recurse=False)]
    
    # Train once with potentially pre-pruned network
    if args['n_rewinds'] == 0:
        prune.global_unstructured(list(zip(p_modules, p_names)), prune.L1Unstructured, amount=args['prune_ratio'])
        train(args['n_epochs'], model, train_dl, valid_dl, device=device)
        return


    # Pretrain epochs to rewinding point
    state_dict = {}
    if args['rewind_to'] > 0:
        train(args['rewind_to'], model, train_dl, valid_dl, state_dict=state_dict, logprefix='Pretrain', device=device)
    
    # Store state to rewind to
    rewind_to_state_dict = copy.deepcopy(state_dict)
    rewind_to_model = copy.deepcopy(model).to('cpu')

    # Train with rewinding and iterative magnitude pruning
    pruner = prune.L1Unstructured(amount=0)                                                             # start with no pruning
    mask = torch.ones_like(torch.cat([p.view(-1) for p in model.parameters()]))                         # and no mask
    for rewind_idx in range(args['n_rewinds']):
        logprefix = f'Rewind{rewind_idx:02d}' if rewind_idx > 0 else ''

        # Training
        best_acc, last_acc = train(args['n_epochs'], model, train_dl, valid_dl, state_dict=state_dict, logprefix=logprefix, device=device)
        wandb_run.log({'IMP/rewind':rewind_idx, 'IMP/ratio':pruner.amount, 
                        'IMP/best_acc':best_acc, 'IMP/last_acc':last_acc, })

        # Compute global mask
        pruner.amount = pruner.amount + (1-pruner.amount) * args['prune_ratio']                         # set cumulative pruning ratio
        [prune.remove(m, n) for m, n in zip(p_modules, p_names) if prune.is_pruned(m)]                  # bake pruning into parameters (remove re-parametrization)
        vec = torch.cat([p.view(-1) for m in model.modules() for p in m.parameters(recurse=False)])     # access parameters() with standard prametrization 
        mask:torch.Tensor = pruner.compute_mask(vec, mask)                                              # this has no side-effects

        # Rewind to init
        state_dict = copy.deepcopy(rewind_to_state_dict)
        model = copy.deepcopy(rewind_to_model).to(device=device)                                        # model now is not 
        p_names = [n for m in model.modules() for n, p in m.named_parameters(recurse=False)]            # store names to restore prametrization
        p_modules = [m for m in model.modules() for p in m.parameters(recurse=False)]                   # store parent modules to restore prametrization

        # Apply global mask to module
        idx = 0  
        for m in model.modules():
            for n, p in list(m.named_parameters(recurse=False)):                                        # list since _parameters odict will change during iteration
                prune.custom_from_mask(m, n, mask[idx:idx+p.numel()].view_as(p))                        # introduce re-parametrization: .name = .name_orig * .name_mask
                idx += p.numel()
        assert idx == mask.numel(), 'Did not iterate over entire mask.'

    return


def train(train_epochs, model:Encoder, train_dl:DataLoader, valid_dl:DataLoader, state_dict:dict=None, logprefix='', device=None) -> float:
    wandb_run:wandb.wandb_sdk.wandb_run.Run = wandb.run # for pylint
    barprefix = '' if (logprefix == '') else f'{logprefix}: '
    logprefix = '' if (logprefix == '') else f'{logprefix}/'
    wandb_run.define_metric(f'{logprefix}train/acc', summary='max')
    wandb_run.define_metric(f'{logprefix}valid/acc', summary='max')

    # Prepare training
    step, epoch = 0, 0
    model.requires_grad_(True)
    optimizer = get_optimizer(args, model.parameters())

    # Setup Random Generators
    seed = torch.seed() if args['seed'] is None else args['seed']
    generator = torch.manual_seed(seed) # set default random generator seed

    # Overwrite state from state_dict, if keys exist exist
    if state_dict is not None:
        step = state_dict.get('step', step)
        epoch = state_dict.get('epoch', epoch)
        optimizer.load_state_dict(state_dict.get('optimizer', optimizer.state_dict()))
        generator.set_state(state_dict.get('generator', generator.get_state()))
    
    # use total n_epochs, since schedules stay the same
    scheduler = get_scheduler(args, optimizer).prep(args['n_epochs'], len(train_dl)) 
    
    # Train Loop
    best_acc = 0
    train_acc = Accuracy(task='multiclass', num_classes=args['n_classes']).to(device)
    valid_acc = Accuracy(task='multiclass', num_classes=args['n_classes']).to(device)
    while epoch < train_epochs:
        
        # Training Epoch
        model.train()
        train_acc.reset()
        progress_bar = tqdm(train_dl, desc=f'{barprefix}Train Epoch {epoch}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

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
            logs = {'trainer/step': step, 'train/acc':acc, 'train/loss':loss,
                        'hparams/lr':optimizer.param_groups[0]['lr'],
                        'hparams/wd':optimizer.param_groups[0]['weight_decay']}
            if step % args['log_every'] == 0:
                wandb_run.log({f'{logprefix}{k}':v for k,v in logs.items()})
            step += 1

        # Validation Epoch
        model.eval()
        valid_acc.reset()
        valid_loss, numel = 0, 0
        progress_bar = tqdm(valid_dl, desc=f'{barprefix}Valid Epoch {epoch}', leave=False)
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
        logs = {'trainer/epoch':epoch, 'trainer/step': step,
                    'valid/loss':valid_loss, 'valid/acc':valid_acc.compute()}
        wandb_run.log({f'{logprefix}{k}':v for k,v in logs.items()})
        best_acc = max(best_acc, valid_acc.compute())
        epoch += 1

    if state_dict is not None:
        state_dict['step'] = step
        state_dict['epoch'] = epoch
        state_dict['optimizer'] = optimizer.state_dict()
        state_dict['generator'] = generator.get_state()
        # scheduler state is recomputed every time

    return best_acc, valid_acc.compute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to finetune.')
    parser.add_argument('--prune_ratio', type=float, default=0.0,
                        help='Ratio of globally smallest weights to prune.')
    parser.add_argument('--n_rewinds', type=int, default=0,
                        help='Number of times to rewind and prune for IMP. Applies pruning before training if =0.')
    parser.add_argument('--rewind_to', type=int, default=0,
                        help='Epoch to rewind to for IMP.')
    
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
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for dataloader and default generator.')

    parser.add_argument('--log_every', type=int, default=1,
                        help='Log every so many steps.')
    parser.add_argument('--force_cpu', action='store_true')
    args = vars(parser.parse_args())

    if args['ckpt'] is None:
        raise ValueError('Please specify checkpoint.')

    if args['rewind_to'] >= args['n_epochs']:
        raise ValueError('Rewinding point must be smaller than n_epochs')

    # if path is not relative, make relative for saving to wandb
    if not args['ckpt'].startswith('DINO'): 
        args['ckpt'] = os.path.relpath(args['ckpt'], os.environ['DINO_RESULTS']) 

    # add is_init boolean for easier handling in wandb
    args['is_init'] = ('init' in args['ckpt'].split()[-1])

    # init wandb
    args['dino_config'] = vars(load_config(os.path.join(os.environ['DINO_RESULTS'], args['ckpt']))) # add dino_config
    wandb.init(project='DINOfinetune', dir=os.environ['DINO_RESULTS'], config=args)
    args['ckpt'] = os.path.join(os.environ['DINO_RESULTS'], args['ckpt']) # make absolute

    main()


