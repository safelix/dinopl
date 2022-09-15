import copy
import os
import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from configuration import CONSTANTS as C, create_dataset, create_multicrop
from configuration import Configuration, create_encoder, create_optimizer
from dinopl import *
from dinopl.probing import LinearProbe, LinearProber
from dinopl.scheduling import Schedule
from dinopl.tracking import (FeatureTracker, HParamTracker, MetricsTracker,
                             ParamTracker, PerCropEntropyTracker, FeatureSaver)
from torchinfo import summary

def main(config:Configuration):
    
    # Fix random seed
    if config.seed is None:
        config.seed = int(time.time())
    pl.seed_everything(config.seed)

    # Logger
    wandb_logger = WandbLogger(
            project='DINO',
            save_dir=C.RESULTS_DIR,
            config=config,
        )

    # store into logging directory
    config.logdir = os.path.join(C.RESULTS_DIR, wandb_logger.name, wandb_logger.version)
    os.makedirs(config.logdir, exist_ok=True)
    config.to_json(os.path.join(config.logdir, 'config.json'))
    print(f'Logging Directory: {config.logdir}')

    # Create Multicrop Specification from name
    config.mc_spec = create_multicrop(config)

    # Standard Augmentations, always work on RGB
    self_trfm = transforms.Compose([ # self-training
                    transforms.Lambda(lambda img: img.convert('RGB')),
                    transforms.ToTensor()
                ])
    eval_trfm = transforms.Compose([ # evaluation
                    transforms.Resize(size=config.mc_spec[0]['out_size']),
                    self_trfm
                ])
    mc = MultiCropAugmentation(config.mc_spec, per_crop_transform=self_trfm)


    # Data Loading
    # TODO: target transform for noisy label?
    DSet = create_dataset(config)
    self_train_set = DSet(root=C.DATA_DIR, train=True, transform=mc)
    self_valid_set = DSet(root=C.DATA_DIR, train=False, transform=mc)
    eval_train_set = DSet(root=C.DATA_DIR, train=True, transform=eval_trfm)
    eval_valid_set = DSet(root=C.DATA_DIR, train=False, transform=eval_trfm)
    print(f'Init {DSet.__name__}: size(train)={len(self_train_set)}, size(valid)={len(self_valid_set)}')

    # Model Setup
    enc = create_encoder(config)
    head = DINOHead(config.embed_dim, config.out_dim, 
            hidden_dims=config.hid_dims, 
            l2bot_dim=config.l2bot_dim, 
            use_bn=config.mlp_bn,
            act_fn=config.mlp_act)
    student = DINOModel(enc, head)

    # initialize teacher with same params as student
    if config.t_init == 'student':
        teacher = copy.deepcopy(student)
    
    # initialize teacher with random parameters
    elif config.t_init == 'random':
        t_enc = create_encoder(config)
        t_head = DINOHead(config.embed_dim, config.out_dim, 
            hidden_dims=config.hid_dims, 
            l2bot_dim=config.l2bot_dim, 
            use_bn=config.mlp_bn,
            act_fn=config.mlp_act)
        teacher = DINOModel(t_enc, t_head)

    # initiallize teacher from checkpoint
    elif config.t_init in ['s_ckpt', 't_ckpt']:
        # requires instanciated model to load state dict
        dino_ckpt = DINO.load_from_checkpoint(config.ckpt_path, mc=mc, student=student, teacher=student)
        teacher = dino_ckpt.student if config.t_init[0]=='s' else dino_ckpt.teacher
    
    else:
        raise RuntimeError(f'Teacher initialization strategy \'{config.t_init}\' not supported.')


    print(f'Created encoder and head:')
    summary(student, depth=4, device='cpu',
            input_size=(config.bs_train, 3, config.mc_spec[0]['out_size'], config.mc_spec[0]['out_size']))

    # DINO Setup
    dino = DINO(mc=mc, student=student, teacher=teacher,
                s_mode = config.s_mode,
                t_mode = config.t_mode,
                t_mom  = Schedule.parse(config.t_mom),
                t_bn_mode = config.t_bn_mode,
                t_eval = config.t_eval,
                t_cmom = Schedule.parse(config.t_cmom),
                s_cmom = Schedule.parse(config.s_cmom),
                t_temp = Schedule.parse(config.t_temp),
                s_temp = Schedule.parse(config.s_temp),
                loss = config.loss,
                loss_pairing = config.loss_pairing,
                opt = create_optimizer(config),
                opt_lr = Schedule.parse(config.opt_lr),
                opt_wd = Schedule.parse(config.opt_wd),
                wn_freeze_epochs=config.wn_freeze_epochs)

    # Tracking Logic    
    callbacks = [
            MetricsTracker(), 
            PerCropEntropyTracker(), 
            FeatureTracker(),
            HParamTracker(),
            ParamTracker(dino.student, dino.teacher, track_init=True),
            ParamTracker(dino.student.head, dino.teacher.head, 'head', True),
            ParamTracker(dino.student.enc, dino.teacher.enc, 'enc', True),
        ]
    
    if len(config.save_features) > 0:
        config.save_features = ['embeddings', 'projections', 'logits'] if 'all' in config.save_features else config.save_features
        callbacks += [FeatureSaver(eval_valid_set, n_imgs=64, features=config.save_features, dir=config.logdir)]

    if config.probe_every > 0 and config.probing_epochs > 0:
        s_probe = LinearProbe(encoder=dino.student.enc,
                                embed_dim=config.embed_dim,
                                n_classes=config.n_classes)
        t_probe = LinearProbe(encoder=dino.teacher.enc,
                                embed_dim=config.embed_dim,
                                n_classes=config.n_classes)
        callbacks += [LinearProber(
                        probe_every = config.probe_every,
                        probing_epochs = config.probing_epochs,
                        probes = dict(student=s_probe, teacher=t_probe),
                        train_set = eval_train_set,
                        valid_set = eval_valid_set,
                        dl_args=dict( # arguments for dataloader
                            batch_size = config.bs_eval,
                            num_workers = config.n_workers,
                            pin_memory = False if config.force_cpu else True)
                    )]
    
    ckpt_callback = ModelCheckpoint(dirpath=config.logdir, monitor='probe/student', mode='max',
                        filename='epoch={epoch}-step={step}-probe_student={valid/loss:.3f}', auto_insert_metric_name=False)

    # Training
    trainer = pl.Trainer(
        # training dynamics
        max_epochs=config.n_epochs,
        gradient_clip_val=config.clip_grad,
        callbacks=callbacks+[ckpt_callback],
        #enable_checkpointing=ckpt_callback,

        # logging
        logger=wandb_logger,
        log_every_n_steps=config.log_every,

        # acceleration
        accelerator='cpu' if config.force_cpu else 'gpu',
        devices=None if config.force_cpu else 1,
        #gpus = [1],
        auto_select_gpus=True,

        # performance
        benchmark=True,
        deterministic=False,

        # debugging
        #limit_train_batches=2,
        #limit_val_batches=2,
        )

    dl_args = dict(
        batch_size = config.bs_train,
        num_workers = config.n_workers,
        pin_memory = False if config.force_cpu else True ) 
    self_train_dl = DataLoader(dataset=self_train_set, **dl_args)
    self_valid_dl = DataLoader(dataset=self_valid_set, **dl_args)

    # log updated config to wandb before training
    wandb_logger.experiment.config.update(config)

    trainer.fit(model=dino, 
                train_dataloaders=self_train_dl,
                val_dataloaders=self_valid_dl)


if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)
    main(config)
