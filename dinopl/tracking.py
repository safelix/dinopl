import os
from typing import Dict

import pytorch_lightning as pl
import torch
import wandb
import numpy as np
from torch.utils import data
import wandb
import pandas as pd

from . import utils as U
from . import DINO

__all__ = [
    'MetricsTracker',
    'PerCropEntropyTracker',
    'FeatureTracker',
    'HParamTracker',
    'ParamTracker',
    'FeatureSaver',
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

def np_histogram(tensor: torch.Tensor, bins=64):
    ndarray = tensor.detach().cpu().numpy()
    range = np.nanmin(ndarray), np.nanmax(ndarray)
    return np.histogram(ndarray, bins=bins, range=range)

class FeatureTracker(pl.Callback):       
    def step(self, prefix, out:Dict[str, torch.Tensor], dino:DINO):
        logs = {}
        prefix += '/feat'
        for n in ['embeddings', 'projections', 'logits']:
            t_x = out['teacher'][n].flatten(0,1)     # consider crops as batches
            s_x = out['student'][n].flatten(0,1)     # consider crops as batches
            s_gx = out['student'][n][:2].flatten(0,1)  # consider crops as batches
            n = n[:5]                       # shortname for plots
            
            for i, x in [('t', t_x), ('s', s_x)]:
                # within batch cosine similarity distance
                cossim = torch.corrcoef(x).nan_to_num() # similarity with anything of norm zero is zero
                cossim_triu = cossim[torch.triu_indices(*cossim.shape, offset=1).unbind()] # upper triangular values
                logs[f'{prefix}/{n}/{i}_x.corr().mean()'] = cossim_triu.mean()
                logs[f'{prefix}/{n}/{i}_x.corr().hist()'] = wandb.Histogram(np_histogram=np_histogram(cossim_triu, 64))
                logs[f'{prefix}/{n}/{i}_x.corr().rank()'] = torch.linalg.matrix_rank(cossim, hermitian=True)
                
                # within batch l2 distance
                l2dist = torch.cdist(x, x)
                l2dist_triu = l2dist[torch.triu_indices(*l2dist.shape, offset=1).unbind()] # upper triangular values
                logs[f'{prefix}/{n}/{i}_x.pdist().mean()'] = l2dist_triu.mean()
                logs[f'{prefix}/{n}/{i}_x.pdist().hist()'] = wandb.Histogram(np_histogram=np_histogram(l2dist_triu, 64))

            # between student and teacher
            l2dist = (t_x - s_gx).norm(dim=-1)
            logs[f'{prefix}/{n}/l2(t_x,s_x).mean()'] = l2dist.mean()
            logs[f'{prefix}/{n}/l2(t_x,s_x).hist()'] = wandb.Histogram(np_histogram=np_histogram(l2dist, 64))

            dot = (t_x * s_gx).sum(-1)
            cossim = dot / (t_x.norm(dim=-1) * s_gx.norm(dim=-1))
            logs[f'{prefix}/{n}/cos(t_x,s_x).mean()'] = cossim.mean()
            logs[f'{prefix}/{n}/cos(t_x,s_x).hist()'] = wandb.Histogram(np_histogram=np_histogram(cossim, 64))

        #logs['trainer/global_step'] = dino.global_step
        #dino.logger.experiment.log(logs)
        dino.logger.log_metrics(logs , step=max(dino.global_step - 1, 0)) # -1 because global step is updated just before on train_batch_end

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
        logs['hparams/s_cmom'] = dino.student.head.cmom
        logs['hparams/t_temp'] = dino.teacher.head.temp
        logs['hparams/s_temp'] = dino.student.head.temp
        logs['hparams/t_cent.norm()'] = dino.teacher.head.cent.norm()
        logs['hparams/t_cent.mean()'] = dino.teacher.head.cent.mean()
        logs['hparams/t_cent'] = wandb.Histogram(np_histogram=np_histogram(dino.teacher.head.cent))
        logs['trainer/global_step'] = dino.global_step
        
        #logs['trainer/global_step'] = dino.global_step
        #dino.logger.experiment.log(logs)
        dino.logger.log_metrics(logs , step=dino.global_step)

class ParamTracker(pl.Callback):
    def __init__(self, student, teacher, name=None, track_init:bool=False) -> None:
        self.student = student
        self.teacher = teacher
        self.track_init = track_init
        self.name = '' if name is None else name + '/'
        
    def on_fit_start(self, *_) -> None:
        if self.track_init:
            self.t_ivec = U.module_to_vector(self.teacher).clone()
            self.s_ivec = U.module_to_vector(self.student).clone()

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
            t_vec = t_vec - self.t_ivec
            s_vec =  s_vec - self.s_ivec

            # log position and angle relative to init
            t_norm = torch.norm(t_vec)
            s_norm = torch.norm(s_vec)
            logs[f'params/{self.name}norm(teach - init)'] = t_norm
            logs[f'params/{self.name}norm(stud - init)'] = s_norm
            logs[f'params/{self.name}cos(teach-init, stud-init)'] = torch.dot(t_vec, s_vec) / (t_norm * s_norm)
            
        dino.log_dict(logs)


class FeatureSaver(pl.Callback):
    def __init__(self, valid_set, n_imgs, dir, features=['embeddings', 'projections', 'logits']) -> None:
        super().__init__()
        self.valid_set = valid_set
        self.n_imgs = n_imgs 
        self.dir = dir
        self.features = features
        self.imgs : torch.Tensor
        self.lbls : torch.Tensor
        
    def on_fit_start(self, _: pl.Trainer, dino: DINO) -> None:
        self.imgs = []
        self.lbls = []
        for img, lbl in data.Subset(self.valid_set, range(0, self.n_imgs)):
            self.imgs.append(img)
            self.lbls.append(torch.tensor([lbl]))
        self.imgs = torch.stack(self.imgs).to(dino.device)
        self.lbls = torch.stack(self.lbls).to(dino.device)

        prefix = os.path.join(self.dir, 'valid', 'feat')
        for n in self.features:
            os.makedirs(os.path.join(os.path.join(prefix, n[:5])), exist_ok=True)

        self.on_train_batch_end(None, dino)

    def on_train_batch_end(self, _:pl.Trainer, dino: DINO, *args) -> None:
        out = {}

        # store training mode and switch to eval
        mode_t = dino.teacher.training
        mode_s = dino.student.training
        dino.eval()

        out['teacher'] = dino.teacher(self.imgs)
        out['student'] = dino.student(self.imgs)

        # restore previous mode
        dino.teacher.train(mode_t)
        dino.student.train(mode_s)

        prefix = os.path.join(self.dir, 'valid', 'feat')
        for n in self.features:
            t_feat = out['teacher'][n].squeeze()
            s_feat = out['student'][n].squeeze()
            n_feat = t_feat.shape[-1]

            columns = ['lbl'] + [f'{n}_{i}' for n in ['t', 's'] for i in range(n_feat)] 
            data = torch.cat((self.lbls, t_feat, s_feat), dim=-1).detach().cpu()

            pd.to_pickle(pd.DataFrame(data, columns=columns), 
                            os.path.join(prefix, n[:5], f'table_{dino.global_step:0>6d}.pckl'))
        return