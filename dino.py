import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from configuration import CONSTANTS as C, create_dataset, create_multicrop
from configuration import Configuration, create_encoder, create_optimizer
from dinopl import *
from dinopl.probing import LinearProbe, LinearProber
from dinopl.tracking import (FeatureTracker, HParamTracker, MetricsTracker,
                             ParamTracker, PerCropEntropyTracker)


def main(config:Configuration):

    # Fix random seed
    if config.seed is None:
        config.seed = int(time.time())
    pl.seed_everything(config.seed)


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
    DSet = create_dataset(config)
    self_train_set = DSet(root=C.DATA_DIR, train=True, transform=mc)
    self_valid_set = DSet(root=C.DATA_DIR, train=False, transform=mc)
    eval_train_set = DSet(root=C.DATA_DIR, train=True, transform=eval_trfm)
    eval_valid_set = DSet(root=C.DATA_DIR, train=False, transform=eval_trfm)
    print(f'Init {type(DSet).__name__}: size(train)={len(self_train_set)}, size(valid)={len(self_valid_set)}')

    # Model Setup
    enc = create_encoder(config)
    head = DINOHead(config.embed_dim, config.out_dim, 
            hidden_dims=config.hid_dims, 
            l2_bottleneck_dim=config.bot_dim, 
            use_bn=config.use_bn,
            act_fn=config.mlp_act)
    model = DINOModel(enc, head)


    # DINO Setup
    dino = DINO(mc=mc, model=model,
                t_mode = config.t_mode,
                t_mom  = config.t_mom,
                t_cmom = config.t_cmom,
                t_temp = config.t_temp,
                s_temp = config.s_temp,
                opt = create_optimizer(config),
                opt_lr = config.opt_lr,
                opt_wd = config.opt_wd)

    # Tracking Logic    
    probing_cb = LinearProber(
        probe_every = config.probe_every,
        probing_epochs = config.probing_epochs,
        # probes
        probes = dict(
            student=LinearProbe(
                encoder=dino.student.enc,
                embed_dim=config.embed_dim,
                n_classes=config.n_classes),
            teacher=LinearProbe(
                encoder=dino.teacher.enc,
                embed_dim=config.embed_dim,
                n_classes=config.n_classes
            )),
        # data loading
        train_set = eval_train_set,
        valid_set = eval_valid_set,
        dl_args=dict(
            batch_size = config.bs_eval,
            num_workers = config.n_workers,
            pin_memory = False if config.force_cpu else True)
    )

    callbacks = [
            MetricsTracker(), 
            PerCropEntropyTracker(), 
            FeatureTracker(),
            HParamTracker(),
            ParamTracker(dino.student, dino.teacher, track_init=True),
            ParamTracker(dino.student.head, dino.teacher.head, 'head', True),
            ParamTracker(dino.student.enc, dino.teacher.enc, 'enc', True),
            probing_cb,
        ]

    # Logger
    wandb_logger = WandbLogger(
            project='DINO',
            save_dir=C.RESULTS_DIR,
            config=config,
        )

    # Training
    trainer = pl.Trainer(
        # training dynamics
        max_epochs=config.n_epochs,
        gradient_clip_val=config.clip_grad,
        callbacks=callbacks,

        # logging
        logger=wandb_logger,
        log_every_n_steps=config.log_every,

        # acceleration
        accelerator='cpu' if config.force_cpu else 'gpu',
        devices=None if config.force_cpu else 1,
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

    trainer.fit(model=dino, 
                train_dataloaders=self_train_dl,
                val_dataloaders=self_valid_dl)


if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)
    main(config)
