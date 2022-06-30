import copy
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import my_utils as U
from dino import DINO


class MetricsTracker(pl.Callback):
    def step(self, prefix, out:Dict[str, torch.Tensor], dino:DINO):
        logs = {}
        logs[f'{prefix}/CE'] = out['CE']
        logs[f'{prefix}/KL'] = out['KL']
        logs[f'{prefix}/H_preds'] = out['H_preds'].mean()
        logs[f'{prefix}/H_targs'] = out['H_targs'].mean()
        dino.log_dict(logs)

    def on_train_batch_end(self, _:pl.Trainer, dino:DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('train', outputs, dino)
    
    def on_validation_batch_end(self, _:pl.Trainer, dino:DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('valid', outputs, dino)


class PerCropEntropyTracker(pl.Callback):
    def step(self, prefix, H_preds:torch.Tensor, H_targs:torch.Tensor, dino:DINO):
        logs = {}
        for crop_name, H_pred in zip(dino.student.crops['name'], H_preds):
            logs[f'{prefix}/H_preds/{crop_name}'] = H_pred.mean() # compute mean of batch for every crop

        for crop_name, H_targ in zip(dino.teacher.crops['name'], H_targs):
            logs[f'{prefix}/H_targs/{crop_name}'] = H_targ.mean() # compute mean of batch for every crop
        dino.log_dict(logs)

    def on_train_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('train', outputs['H_preds'], outputs['H_targs'], dino)
    
    def on_validation_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('valid', outputs['H_preds'], outputs['H_targs'], dino)


class HParamTracker(pl.Callback):  
    def step(self, dino:DINO):
        logs = {}
        logs['hparams/lr'] = dino.optimizer.param_groups[0]['lr']
        logs['hparams/wd'] = dino.optimizer.param_groups[1]['weight_decay']
        logs['hparams/t_mom'] = dino.t_updater.mom
        logs['hparams/t_cmom'] = dino.teacher.head.cmom
        logs['hparams/t_temp'] = dino.teacher.head.temp
        logs['hparams/s_temp'] = dino.student.head.temp
        logs['hparams/t_cent.norm()'] = dino.teacher.head.cent.norm()
        logs['hparams/t_cent.mean()'] = dino.teacher.head.cent.mean()
        dino.log_dict(logs)

    def on_train_batch_start(self, _:pl.Trainer, dino:DINO, *args) -> None:
        self.step(dino)


class ParamTracker(pl.Callback):
    def __init__(self, track_init:bool=False) -> None:
        self.track_init = track_init
        
    def step(self, dino:DINO):
        logs = {}

        # get vector representation of student and teacher
        t_vec = U.module_to_vector(dino.teacher)
        s_vec = U.module_to_vector(dino.student)

        # get driving signals: gradient and difference
        g_vec = U.module_to_vector(dino.student, grad=True)
        d_vec = t_vec - s_vec

        # log driving signals
        g_norm = torch.norm(g_vec)
        d_norm = torch.norm(d_vec)
        logs['params/norm(grad)'] = g_norm
        logs['params/norm(diff)'] = d_norm
        logs['params/cos(grad, diff)'] = torch.dot(g_vec, d_vec) / (g_norm * d_norm)

        if self.track_init:           
            # get vector representations relative to init
            t_vec = t_vec - self.i_vec
            s_vec =  s_vec - self.i_vec

            # log position and angle relative to init
            t_norm = torch.norm(t_vec)
            s_norm = torch.norm(s_vec)
            logs['params/norm(teach - init)'] = t_norm
            logs['params/norm(stud - init)'] = s_norm
            logs['params/cos(teach-init, stud-init)'] = torch.dot(t_vec, s_vec) / (t_norm * s_norm)
            
        dino.log_dict(logs)

    def on_fit_start(self, _:pl.Trainer, dino: DINO, *args) -> None:
        if self.track_init:
            self.i_vec = U.module_to_vector(dino.teacher).clone()

        #if isinstance(dino.logger, WandbLogger):
        #    dino.logger.watch(dino.student, log='all', log_freq=100)

    def on_train_batch_end(self, _:pl.Trainer, dino: DINO, *args) -> None:
        self.step(dino)
