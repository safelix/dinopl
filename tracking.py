import copy
from typing import Dict, List

import pytorch_lightning as pl
import torch

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
    
    def on_valid_batch_end(self, _:pl.Trainer, dino:DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
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
    
    def on_valid_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
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
        dino.log_dict(logs)

    def on_train_batch_start(self, _:pl.Trainer, dino:DINO, *args) -> None:
        self.step(dino)


class ParamTracker(pl.Callback):
    def __init__(self, track_init:bool=False) -> None:
        self.track_init = track_init
        
    def step(self, dino:DINO):
        logs = {}
        s_vec = U.module_to_vector(dino.student)
        t_vec = U.module_to_vector(dino.teacher)
        logs['params/cos(teacher,student)']  = torch.dot(t_vec, s_vec) / torch.norm(t_vec) / torch.norm(s_vec)
        logs['params/norm(teacher-student)'] = torch.norm(t_vec - s_vec)

        if self.init:
            i_vec = U.module_to_vector(self.init)
            logs['params/cos(init,teacher)']     = torch.dot(i_vec, t_vec) / torch.norm(i_vec) / torch.norm(t_vec)
            logs['params/cos(init,student)']     = torch.dot(i_vec, s_vec) / torch.norm(i_vec) / torch.norm(s_vec)
            logs['params/norm(init-teacher)']    = torch.norm(i_vec - t_vec)
            logs['params/norm(init-student)']    = torch.norm(i_vec - s_vec)

        dino.log_dict(logs)

    def on_fit_start(self, _:pl.Trainer, dino: DINO, *args) -> None:
        self.init = copy.deepcopy(dino.student) if self.track_init else None

    def on_train_batch_end(self, _:pl.Trainer, dino: DINO, *args) -> None:
        self.step(dino)
