import copy
import os
import time
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms

from configuration import CONSTANTS as C
from configuration import (Configuration, get_dataset, create_mc_spec,
                           create_optimizer, get_encoder)
from dinopl import *
from dinopl import utils as U
from dinopl.augmentation import LabelNoiseWrapper, LogitNoiseWrapper, MultiCrop
from dinopl.probing import LinearProbe, LinearProber
from dinopl.scheduling import Schedule
from dinopl.tracking import (FeatureSaver, FeatureTracker, HParamTracker,
                             MetricsTracker, ParamTracker,
                             PerCropEntropyTracker, AccuracyTracker)


def main(config:Configuration):
    
    if config.float64:
        torch.set_default_dtype(torch.float64)
    
    # Fix random seeds
    if config.seed is None:
        config.seed = int(time.time())
    pl.seed_everything(config.seed)

    if config.enc_seed is None:
        config.enc_seed = int(time.time())
    generator = torch.Generator().manual_seed(config.enc_seed)

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
    config.mc_spec = create_mc_spec(config)

    # Standard Augmentations, always work on RGB
    DSet = get_dataset(config)
    self_trfm = transforms.Compose([ # self-training
                    transforms.Lambda(lambda img: img.convert('RGB')), transforms.ToTensor(),
                    transforms.Normalize(DSet.mean, DSet.std),
                ])
    eval_trfm = transforms.Compose([ # evaluation
                    transforms.Resize(size=config.mc_spec[0]['out_size']),
                    self_trfm
                ])
    mc = MultiCrop(config.mc_spec, per_crop_transform=self_trfm)
          
    # Data Setup.
    dino_train_set = DSet(root=C.DATA_DIR, train=True, transform=mc)
    dino_valid_set = DSet(root=C.DATA_DIR, train=False, transform=mc)
    probe_train_set = DSet(root=C.DATA_DIR, train=True, transform=eval_trfm)
    probe_valid_set = DSet(root=C.DATA_DIR, train=False, transform=eval_trfm)

    if config.label_noise_ratio > 0 and config.logit_noise_temp > 0:
        raise RuntimeError('Only either label noise or logit noise can be applied.')
    elif config.label_noise_ratio > 0 and config.ds_classes != config.n_classes:
        raise RuntimeError('Cannot change number of classes with label noise.')
    elif config.label_noise_ratio > 0 and config.ds_classes == config.n_classes:
        dino_train_set = LabelNoiseWrapper(dino_train_set, config.n_classes, config.label_noise_ratio, config.resample_noise)
        #dino_valid_set = LabelNoiseWrapper(dino_valid_set, config.n_classes, config.label_noise_ratio, config.resample_noise)
    elif config.logit_noise_temp > 0:
        dino_train_set = LogitNoiseWrapper(dino_train_set, config.n_classes, config.logit_noise_temp, config.resample_noise)
        dino_valid_set = LogitNoiseWrapper(dino_valid_set, config.n_classes, config.logit_noise_temp, config.resample_noise)
    
    print(f'Init dino train set: {dino_train_set}')
    print(f'Init dino valid set: {dino_valid_set}')

    dl_args = dict(
        num_workers = config.n_workers,
        pin_memory = False if config.force_cpu else True) 
    dino_train_dl = DataLoader(dataset=dino_train_set, batch_size=config.bs_train, shuffle=True, generator=generator, **dl_args)
    dino_valid_dl = DataLoader(dataset=dino_valid_set, batch_size=config.bs_eval, **dl_args)
    probe_train_dl = DataLoader(dataset=probe_train_set, batch_size=config.bs_train, shuffle=True, generator=torch.Generator(), **dl_args)
    probe_valid_dl = DataLoader(dataset=probe_valid_set, batch_size=config.bs_eval, **dl_args)

    # Model Setup.
    enc = get_encoder(config)()
    config.embed_dim = enc.embed_dim
    head = DINOHead(config.embed_dim, config.out_dim, 
            hidden_dims=config.hid_dims, 
            l2bot_dim=config.l2bot_dim, 
            l2bot_cfg=config.l2bot_cfg,
            use_bn=config.mlp_bn,
            act_fn=config.mlp_act)
    model = DINOModel(enc, head)
    model.reset_parameters(generator=generator)

    print(f'Created encoder and head:')
    summary(model, depth=4, device='cpu',
            input_size=(config.bs_train, 3, config.mc_spec[0]['out_size'], config.mc_spec[0]['out_size']))

    # load checkpoint if required
    if config.s_init in ['s_ckpt', 't_ckpt'] or config.t_init in ['s_ckpt', 't_ckpt']:
        if getattr(config, 'ckpt_path', '') == '':
            raise RuntimeError('Student or teacher inititalization strategy requires \'--ckpt_path\' to be specified.')
        temp_student = copy.deepcopy(model) # required to load state dict into instanciated copy
        temp_teacher = copy.deepcopy(model) # required to load state dict into instanciated copy
        dino_ckpt = DINO.load_from_checkpoint(config.ckpt_path,  mc_spec=config.mc_spec, student=temp_student, teacher=temp_teacher)
        del temp_student, temp_teacher
    
    # Initialize student network
    if config.s_init == 'random':
        student = copy.deepcopy(model)  # make student with random params
    elif config.s_init == 's_ckpt':
        student = copy.deepcopy(dino_ckpt.student)     # make student from student checkpoint
    elif config.s_init == 't_ckpt':
        student = copy.deepcopy(dino_ckpt.teacher)     # make student from teacher checkpoint
    else:
        raise RuntimeError(f'Student initialization strategy \'{config.t_init}\' not supported.')

    # Initialize teacher network
    if config.t_init == 'student':
        teacher = copy.deepcopy(student) # make teacher with same params as student
    elif config.s_init == 's_ckpt':
        teacher = copy.deepcopy(dino_ckpt.student)     # make teacher from student checkpoint
    elif config.s_init == 't_ckpt':
        teacher = copy.deepcopy(dino_ckpt.teacher)     # make teacher from teacher checkpoint
    elif config.t_init == 'random':     
        teacher = copy.deepcopy(student)
        teacher.reset_parameters(generator=generator)  # initialize teacher with random parameters
    elif config.t_init == 'interpolated':
        teacher = copy.deepcopy(student)
        teacher.reset_parameters(generator=generator)  # initialize teacher with random parameters
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            alpha = config.t_init_alpha
            p_t.data = (1 - alpha) * p_s + alpha * p_t # interpolate between random and student
            if config.t_init_var_preserving:
                p_t.data /= math.sqrt(2*alpha**2  - 2*alpha + 1)   # apply variance preserving correction
    elif config.t_init == 'neighborhood':
        teacher = copy.deepcopy(student)
        teacher.reset_parameters(generator=generator)  # initialize teacher with random parameters
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            eps = config.t_init_eps
            p_t.data = p_s + eps * p_t # add eps neighborhood to student
            if config.t_init_var_preserving:
                p_t.data /= math.sqrt(eps**2 + 1)   # apply variance preserving correction
    else:
        raise RuntimeError(f'Teacher initialization strategy \'{config.t_init}\' not supported.')
    del model #, dino_ckpt # let's hope for garbage collector


    # DINO Setup
    dino = DINO(mc_spec=config.mc_spec, student=student, teacher=teacher,
                s_mode = config.s_mode,
                t_mode = config.t_mode,
                t_mom  = Schedule.parse(config.t_mom),
                t_update_every = config.t_update_every,
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
            AccuracyTracker(supervised=(config.s_mode=='supervised'), 
                            logit_targets=(config.logit_noise_temp > 0))
        ]
    wandb_logger.experiment.define_metric('train/s_acc', summary='max')
    wandb_logger.experiment.define_metric('valid/s_acc', summary='max')
    

    if len(config.save_features) > 0:
        config.save_features = ['embeddings', 'projections', 'logits'] if 'all' in config.save_features else config.save_features
        callbacks += [FeatureSaver(probe_valid_set, n_imgs=64, features=config.save_features, dir=config.logdir)]

    if config.probe_every > 0 and config.probing_epochs > 0:
        s_probe = LinearProbe(encoder=dino.student.enc,
                                embed_dim=config.embed_dim,
                                n_classes=config.ds_classes)
        t_probe = LinearProbe(encoder=dino.teacher.enc,
                                embed_dim=config.embed_dim,
                                n_classes=config.ds_classes)
        callbacks += [LinearProber(
                        probe_every = config.probe_every,
                        probing_epochs = config.probing_epochs,
                        probes = dict(student=s_probe, teacher=t_probe),
                        train_dl = probe_train_dl,
                        valid_dl = probe_valid_dl,
                        seed=config.prober_seed
                    )]
        wandb_logger.experiment.define_metric('probe/student', summary='max')
        wandb_logger.experiment.define_metric('probe/teacher', summary='max')


    ckpt_callback = ModelCheckpoint(dirpath=config.logdir, monitor='probe/student', mode='max',
                        filename='epoch={epoch}-step={step}-probe_student={probe/student:.3f}', auto_insert_metric_name=False)

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
        num_sanity_val_steps=0, # call trainer.validate() before trainer.fit() instead

        # acceleration
        accelerator='cpu' if config.force_cpu else 'gpu',
        devices=None if config.force_cpu else [U.pick_single_gpu()],
        auto_select_gpus=False,

        # performance
        benchmark=True,
        deterministic=False,

        # debugging
        #limit_train_batches=2,
        #limit_val_batches=2,
        )

    # log updated config to wandb before training
    wandb_logger.experiment.config.update(config, allow_val_change=True)

    # move dino to selected GPU, validate, then fit
    dino = dino if config.force_cpu else dino.to(trainer.device_ids[0])
    trainer.validate(model=dino, dataloaders=dino_valid_dl)
    trainer.fit(model=dino, 
                train_dataloaders=dino_train_dl,
                val_dataloaders=dino_valid_dl)


if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)
    main(config)
