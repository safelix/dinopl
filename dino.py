import copy
import math
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms

from configuration import CONSTANTS as C
from configuration import (Configuration, create_mc_spec, create_optimizer,
                           get_dataset, get_encoder, init_student_teacher, get_augmentations)
from datasets.targetnoise import LabelNoiseWrapper, LogitNoiseWrapper
from dinopl import *
from dinopl import utils as U
from dinopl.probing import KNNAnalysis, LinearAnalysis, Prober
from dinopl.scheduling import Schedule
from dinopl.tracking import (AccuracyTracker, FeatureSaver, FeatureTracker,
                             HParamTracker, MetricsTracker, ParamStatSaver,
                             ParamTracker, GradVarTracker, PerCropEntropyTracker)


def main(config:Configuration):
    devices = None if config.force_cpu else [U.pick_single_gpu()]

    if config.float64:
        torch.set_default_dtype(torch.float64)
    
    # Fix random seeds
    if config.seed is None:
        config.seed = int(time.time())
    pl.seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

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
    DSet = get_dataset(config)

    # Standard Tranformations: make tensor, normalize, cast to float64 if needed
    trfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(DSet.mean, DSet.std)])
    if config.float64: 
        trfm = transforms.Compose([trfm, transforms.ConvertImageDtype(torch.float64)])

    # Add DINO transformations
    dino_trfm = transforms.Compose([
        get_augmentations(config, DSet), # first apply dataset specific augmentations (for all imgs)
        MultiCrop(config.mc_spec, per_crop_transform=transforms.Compose([ # make crops
            get_augmentations(config, DSet, per_crop=True), # make per crop augmentations
            trfm # standard transformations are applied at the end per crop
        ]))
    ])

    # Resize to first cropsize of multicrop, apply standard transformations 
    eval_trfm = transforms.Compose([transforms.Resize(size=config.mc_spec[0]['out_size']), trfm])
          
    # Data Setup.
    dino_train_set = DSet(root=C.DATA_DIR, train=True, transform=dino_trfm)
    dino_valid_set = DSet(root=C.DATA_DIR, train=False, transform=dino_trfm)
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
    drop_last = getattr(config, 'batchaccum', None) is not None # drop last in case of gradient accumulation

    dino_train_dl = DataLoader(dataset=dino_train_set, batch_size=config.bs_train, shuffle=True, generator=generator, drop_last=drop_last, **dl_args)
    dino_valid_dl = DataLoader(dataset=dino_valid_set, batch_size=config.bs_eval, **dl_args)
    probe_train_dl = DataLoader(dataset=probe_train_set, batch_size=config.bs_eval, shuffle=True, generator=torch.Generator(), **dl_args)
    probe_valid_dl = DataLoader(dataset=probe_valid_set, batch_size=config.bs_eval, **dl_args)
    # -1 is full batch gradient descent
    if getattr(config, 'batchaccum', None) == -1:
        config.batchaccum = len(dino_train_dl) 

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

    print(f'Created encoder and head:')
    summary(model, depth=4, device='cpu', input_data=[next(iter(dino_valid_dl))[0]])

    student, teacher = init_student_teacher(config=config, model=model)
    del model # don't need this anymore


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
    print(f'Init optimizer: {len(dino.optimizer.param_groups)} paramgroups of sizes', 
        [len(group['params']) for group in dino.optimizer.param_groups])
    print(f'=> {dino.optimizer}')

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
    

    if config.probe_every > 0:
        analyses = {}
        if config.probing_epochs > 0:
            analyses[''] = LinearAnalysis(config.probing_epochs)
            wandb_logger.experiment.define_metric('probe/student', summary='max')
            wandb_logger.experiment.define_metric('probe/teacher', summary='max')

        if config.probing_k > 0:
            analyses['knn'] = KNNAnalysis(config.probing_k)
            wandb_logger.experiment.define_metric('probe/student/knn', summary='max')
            wandb_logger.experiment.define_metric('probe/teacher/knn', summary='max')
        
        encoders = dict(student=dino.student.enc, teacher=dino.teacher.enc)
        callbacks += [Prober(encoders=encoders, analyses=analyses, 
                                train_dl = probe_train_dl,
                                valid_dl = probe_valid_dl,
                                n_classes = config.ds_classes,
                                normalize = config.normalize_probe,
                                probe_every = config.probe_every,
                                seed = config.prober_seed
                            )]

    if config.track_gradvar:
        model = dino.student
        callbacks += [GradVarTracker(model, {'enc':model.enc, 'head':model.head})]  

    if len(config.save_features) > 0:
        config.save_features = ['embeddings', 'projections', 'logits'] if 'all' in config.save_features else config.save_features
        callbacks += [FeatureSaver(probe_valid_set, n_imgs=64, features=config.save_features, dir=config.logdir)]

    if len(config.save_paramstats) > 0:
        config.save_paramstats = ['teacher', 'student'] if 'all' in config.save_paramstats else config.save_paramstats
        if 'teacher' in config.save_paramstats:
            callbacks += [ParamStatSaver(dino.teacher, 'teacher', dir=config.logdir)]
        if 'student' in config.save_paramstats:
            callbacks += [ParamStatSaver(dino.student, 'student', dir=config.logdir)]
        

    ckpt_callback = ModelCheckpoint(dirpath=config.logdir, monitor='probe/student', mode='max', save_last=True,
                        filename='epoch={epoch}-step={step}-probe_student={probe/student:.3f}', auto_insert_metric_name=False)

    # Training
    trainer = pl.Trainer(
        # training dynamics
        max_epochs=config.n_epochs,
        max_steps=config.n_steps,
        gradient_clip_val=config.clip_grad,
        callbacks=callbacks+[ckpt_callback],
        #enable_checkpointing=ckpt_callback,
        accumulate_grad_batches=getattr(config, 'batchaccum', None),

        # logging
        logger=wandb_logger,
        log_every_n_steps=config.log_every,
        num_sanity_val_steps=0, # call trainer.validate() before trainer.fit() instead

        # acceleration
        accelerator='cpu' if config.force_cpu else 'gpu',
        devices=devices,
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
    config.to_json(os.path.join(config.logdir, 'config.json'))

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
