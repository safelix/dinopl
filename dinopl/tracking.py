from typing import Dict

import pytorch_lightning as pl
import torch

import dinopl.utils as U
from dinopl import DINO

__all__ = [
    'MetricsTracker',
    'PerCropEntropyTracker',
    'FeatureTracker',
    'HParamTracker',
    'ParamTracker',
]

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
    def step(self, prefix, out:Dict[str, torch.Tensor], dino:DINO):
        logs = {}
        for crop_name, H_pred in zip(dino.student.crops['name'], out['H_preds']):
            logs[f'{prefix}/H_preds/{crop_name}'] = H_pred.mean() # compute mean of batch for every crop

        for crop_name, H_targ in zip(dino.teacher.crops['name'], out['H_targs']):
            logs[f'{prefix}/H_targs/{crop_name}'] = H_targ.mean() # compute mean of batch for every crop
        dino.log_dict(logs)

    def on_train_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('train', outputs, dino)
    
    def on_validation_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('valid', outputs, dino)

class FeatureTracker(pl.Callback):       
    def step(self, prefix, out:Dict[str, torch.Tensor], dino:DINO):
        logs = {}
        prefix += '/feat'
        for n in ['embeddings', 'projections', 'logits']:
            t_x = out['teacher'][n].flatten(0,1)   # consider crops as batches
            s_x = out['student'][n].flatten(0,1)   # consider crops as batches
            s_gx = out['student'][n][:2].flatten(0,1)   # consider crops as batches
            n = n[:5]                       # shortname for plots
            
            for i, x in [('t', t_x), ('s', s_x)]:
                # within batch cosine similarity distance
                cossim = torch.corrcoef(x).triu(diagonal=1) # upper triangular
                logs[f'{prefix}/{n}/{i}_x.corr().mean()'] = cossim.mean()
                
                # within batch l2 distance
                l2dist = torch.cdist(x, x).triu(diagonal=1) # upper triangular
                logs[f'{prefix}/{n}/{i}_x.pdist().mean()'] = l2dist.mean()

            # between student and teacher
            l2dist = (t_x - s_gx).norm(dim=-1)
            logs[f'{prefix}/{n}/l2(t_x,s_x).mean()'] = l2dist.mean()

            dot = (t_x * s_gx).sum(-1)
            cossim = dot / (t_x.norm(dim=-1) * s_gx.norm(dim=-1))
            logs[f'{prefix}/{n}/cos(t_x,s_x).mean()'] = cossim.mean()

        dino.log_dict(logs)

    def on_train_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('train', outputs, dino)

    def on_validation_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('valid', outputs, dino)

class HParamTracker(pl.Callback):  
    def on_train_batch_start(self, _:pl.Trainer, dino:DINO, *args) -> None:
        logs = {}
        logs['hparams/lr'] = dino.optimizer.param_groups[0]['lr']
        logs['hparams/wd'] = dino.optimizer.param_groups[0]['weight_decay']
        logs['hparams/t_mom'] = dino.t_updater.mom
        logs['hparams/t_cmom'] = dino.teacher.head.cmom
        logs['hparams/t_temp'] = dino.teacher.head.temp
        logs['hparams/s_temp'] = dino.student.head.temp
        logs['hparams/t_cent.norm()'] = dino.teacher.head.cent.norm()
        logs['hparams/t_cent.mean()'] = dino.teacher.head.cent.mean()
        dino.log_dict(logs)

class ParamTracker(pl.Callback):
    def __init__(self, student, teacher, name=None, track_init:bool=False) -> None:
        self.student = student
        self.teacher = teacher
        self.track_init = track_init
        self.name = '' if name is None else name + '/'
        
    def on_fit_start(self, *_) -> None:
        if self.track_init:
            self.i_vec = U.module_to_vector(self.teacher).clone()

    def on_after_backward(self, *_) -> None:
        self.g_vec = U.module_to_vector(self.student, grad=True)

    def on_train_batch_end(self, _:pl.Trainer, dino: DINO, *args) -> None:
        logs = {}

        # get vector representation of student and teacher
        t_vec = U.module_to_vector(self.teacher)
        s_vec = U.module_to_vector(self.student)

        # get driving signals: gradient and difference
        g_vec = self.g_vec #U.module_to_vector(self.student, grad=True)
        d_vec = t_vec - s_vec

        # log driving signals
        g_norm = torch.norm(g_vec)
        d_norm = torch.norm(d_vec)
        logs[f'params/{self.name}norm(grad)'] = g_norm
        logs[f'params/{self.name}norm(diff)'] = d_norm
        logs[f'params/{self.name}cos(grad, diff)'] = torch.dot(g_vec, d_vec) / (g_norm * d_norm)

        if self.track_init:           
            # get vector representations relative to init
            t_vec = t_vec - self.i_vec
            s_vec =  s_vec - self.i_vec

            # log position and angle relative to init
            t_norm = torch.norm(t_vec)
            s_norm = torch.norm(s_vec)
            logs[f'params/{self.name}norm(teach - init)'] = t_norm
            logs[f'params/{self.name}norm(stud - init)'] = s_norm
            logs[f'params/{self.name}cos(teach-init, stud-init)'] = torch.dot(t_vec, s_vec) / (t_norm * s_norm)
            
        dino.log_dict(logs)
