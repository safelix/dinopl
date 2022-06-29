import copy
from typing import Dict

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

        # get vector representation of student and teacher
        t_vec = U.module_to_vector(dino.teacher)
        s_vec = U.module_to_vector(dino.student)

        # log absolute vector representations
        t_norm = torch.norm(t_vec)
        s_norm = torch.norm(s_vec)
        logs['params/norm(teach)'] = t_norm
        logs['params/norm(stud)'] = s_norm
        logs['params/cos(teach, stud)']  = torch.dot(t_vec, s_vec) / (t_norm * s_norm)
        
        # get driving signals: gradient and difference
        g_vec = U.module_to_vector(dino.student, grad=True)
        d_vec = t_vec - s_vec

        # log driving signals
        g_norm = torch.norm(g_vec)
        d_norm = torch.norm(d_vec)
        logs['params/norm(grad)'] = g_norm
        logs['params/norm(diff)'] = d_norm
        logs['params/cos(grad, diff)'] = torch.dot(g_vec, d_vec) / (g_norm * d_norm)

        if self.init:
            # get absolute vector representations of init
            i_vec = U.module_to_vector(self.init)
            i_norm = torch.norm(i_vec)
            logs['params/cos(teacher, init)']  = torch.dot(t_vec, i_vec) / (t_norm * i_norm)
            logs['params/cos(student, init)']  = torch.dot(s_vec, i_vec) / (s_norm * i_norm)
            
            # get vector representations relative to init
            t_vec, s_vec = t_vec - i_vec, s_vec - i_vec
            t_norm = torch.norm(t_vec)
            s_norm = torch.norm(s_vec)
            logs['params/norm(teach - init)'] = t_norm
            logs['params/norm(stud - init)'] = s_norm
            logs['params/cos(teach-init, stud-init)'] = torch.dot(t_vec, s_vec) / (t_norm * s_norm)
            
        dino.log_dict(logs)

    def on_fit_start(self, _:pl.Trainer, dino: DINO, *args) -> None:
        self.init = copy.deepcopy(dino.student) if self.track_init else None

    def on_train_batch_end(self, _:pl.Trainer, dino: DINO, *args) -> None:
        self.step(dino)
