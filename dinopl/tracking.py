import os
from typing import Any, Dict, List, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from torch.nn import functional as F
from torch.utils import data
from torchmetrics import Accuracy

from . import DINO
from . import utils as U

__all__ = [
    'AccuracyTracker'
    'MetricsTracker',
    'PerCropEntropyTracker',
    'FeatureTracker',
    'FeatureHistTracker',
    'HParamTracker',
    'ParamTracker',
    'FeatureSaver',
]

# Callbacks/Hooks and when they are called:
# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

class AccuracyTracker(pl.Callback):
    def __init__(self, n_classes, supervised=True, logit_targets=False) -> None:
        self.s_train_acc = Accuracy(task='multiclass', num_classes=n_classes)
        self.s_valid_acc = Accuracy(task='multiclass', num_classes=n_classes)
        self.supervised = supervised
        self.logit_targets = logit_targets

    
    def on_train_batch_end(self, _: pl.Trainer, dino:DINO, out, batch, *args):
        batch, targets = batch
        self.s_train_acc.to(dino.device)

        if not self.supervised: # teacher logits -> probas -> mean over crops -> labels 
            targets = F.softmax(out['teacher']['logits'], dim=-1).mean(dim=0).argmax(dim=-1)

        if self.supervised and self.logit_targets: # gaussian logits -> labels
            targets = F.softmax(targets, dim=-1).argmax(dim=-1)

        # compute average probabilities over all crops, acc takes logits or probas
        probas = F.softmax(out['student']['logits'], dim=-1).mean(dim=0)
        s_acc = self.s_train_acc(probas, targets)
        dino.log('train/s_acc', s_acc, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, *args):
        self.s_train_acc.reset()

    def on_validation_batch_end(self, _: pl.Trainer, dino:DINO, out, batch, *args):
        batch, targets = batch
        self.s_valid_acc.to(dino.device)

        if not self.supervised: # teacher logits -> probas -> mean over crops -> labels 
            targets = F.softmax(out['teacher']['logits'], dim=-1).mean(dim=0).argmax(dim=-1)

        if self.supervised and self.logit_targets: # gaussian logits -> labels
            targets = F.softmax(targets, dim=-1).argmax(dim=-1)
        
        # compute average probabilities over all crops, acc takes logits or probas
        probas = F.softmax(out['student']['logits'], dim=-1).mean(dim=0)
        self.s_valid_acc.update(probas, targets)
    
    @torch.inference_mode(False) # reset needs to be done outside of InferenceMode
    def on_validation_epoch_end(self, _: pl.Trainer, dino:DINO, *args):
        dino.log('valid/s_acc', self.s_valid_acc.compute(), on_step=False, on_epoch=True)
        self.s_valid_acc.reset()


class MetricsTracker(pl.Callback):
    def step(self, prefix, out:Dict[str, torch.Tensor], dino:DINO):
        logs = {}
        logs[f'{prefix}/loss'] = out['loss']
        if dino.loss in ['CE', 'KL', 'H_preds']:
            logs[f'{prefix}/CE'] = out['CE']
            logs[f'{prefix}/KL'] = out['KL']
            logs[f'{prefix}/H_preds'] = out['H_preds'].mean()
            logs[f'{prefix}/H_targs'] = out['H_targs'].mean()
        elif dino.loss in ['MSE']:
            logs[f'{prefix}/MSE'] = out['MSE']

        dino.log_dict(logs)

    def on_train_batch_end(self, _:pl.Trainer, dino:DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('train', outputs, dino)
    
    def on_validation_batch_end(self, _:pl.Trainer, dino:DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('valid', outputs, dino)


class PerCropEntropyTracker(pl.Callback):
    def step(self, prefix, out:Dict[str, torch.Tensor], dino:DINO):
        if dino.loss not in ['CE', 'KL', 'H_preds']:
            return
            
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

def wandb_histogram(tensor: torch.Tensor, bins=64):
    try:
        ndarray = tensor.detach().cpu().numpy()
        range = np.nanmin(ndarray), np.nanmax(ndarray)
        return wandb.Histogram(np_histogram=np.histogram(ndarray, bins=bins, range=range))
    except Exception as e:
        warn(f'Cannot create histogram ({str(e)}), returning None.')
        return None

def matrix_pca(matrix:torch.Tensor):
    try:
        return torch.linalg.svdvals(matrix - matrix.mean(dim=0))
    except Exception as e:
        warn(f'Cannot compute svd ({str(e)}), returning None')
        return None

def matrix_rank(matrix, hermitian=False):
    try:
        rank = torch.linalg.matrix_rank(matrix, hermitian=hermitian)
        return float(rank)
    except Exception as e:
        warn(f'Cannot compute rank ({str(e)}), returning torch.nan.')
        return torch.nan

def batch_cossim(x:torch.Tensor):
    try:
        cossim = torch.corrcoef(x).nan_to_num() # similarity with anything of norm zero is zero
        cossim = cossim.reshape(x.shape[0], x.shape[0]) # incase dimension is lost
        return cossim
    except Exception as e:
        warn(f'Cannot batch cosssim ({str(e)}), returning torch.nan.')
        return torch.full((x.shape[0], x.shape[0]), torch.nan, dtype=x.dtype, device=x.device)

def batch_l2dist(x:torch.Tensor):
    try:
        l2dist = torch.cdist(x,x)
        l2dist = l2dist.reshape(x.shape[0], x.shape[0]) # incase dimension is lost
        return l2dist
    except Exception as e:
        warn(f'Cannot batch l2dist ({str(e)}), returning torch.nan.')
        return torch.full((x.shape[0], x.shape[0]), torch.nan, dtype=x.dtype, device=x.device)

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
                logs[f'{prefix}/{n}/{i}_x.mean().mean()'] = x.mean(dim=0).mean()
                logs[f'{prefix}/{n}/{i}_x.mean().std()'] = x.mean(dim=0).std()
                logs[f'{prefix}/{n}/{i}_x.std().mean()'] = x.std(dim=0).mean()
                logs[f'{prefix}/{n}/{i}_x.std().std()'] = x.std(dim=0).std()
                logs[f'{prefix}/{n}/{i}_x.rank()'] = matrix_rank(x)

                pc_sigma = matrix_pca(x)
                if pc_sigma is not None:
                    logs[f'{prefix}/{n}/{i}_x.nPC(var_explained=0.95)'] = float(pc_sigma.numel() - torch.sum(0.95 < torch.cumsum(pc_sigma, dim=0) / pc_sigma.sum()))
                    logs[f'{prefix}/{n}/{i}_x.nPC(var_explained=0.99)'] = float(pc_sigma.numel() - torch.sum(0.99 < torch.cumsum(pc_sigma, dim=0) / pc_sigma.sum()))
                    logs[f'{prefix}/{n}/{i}_x.nPC(var_explained=0.999)'] = float(pc_sigma.numel() - torch.sum(0.999 < torch.cumsum(pc_sigma, dim=0) / pc_sigma.sum()))

                # within batch cosine similarity distance
                cossim = batch_cossim(x)
                cossim_triu = cossim[torch.triu_indices(*cossim.shape, offset=1).unbind()] # upper triangular values
                logs[f'{prefix}/{n}/{i}_x.corr().mean()'] = cossim_triu.mean()
                logs[f'{prefix}/{n}/{i}_x.corr().rank()'] = matrix_rank(cossim, hermitian=False) # TODO: hermitian=True breaks with torch 2.0.1+cu117
                
                # within batch l2 distance
                l2dist = batch_l2dist(x)
                l2dist_triu = l2dist[torch.triu_indices(*l2dist.shape, offset=1).unbind()] # upper triangular values
                logs[f'{prefix}/{n}/{i}_x.pdist().mean()'] = l2dist_triu.mean()

            # between student and teacher
            l2dist = (t_x - s_gx).norm(dim=-1)
            logs[f'{prefix}/{n}/l2(t_x,s_x).mean()'] = l2dist.mean()

            rmse = (t_x - s_gx).square().mean(dim=-1).sqrt()
            logs[f'{prefix}/{n}/rmse(t_x,s_x).mean()'] = rmse.mean()

            dot = (t_x * s_gx).sum(-1)
            cossim = dot / (t_x.norm(dim=-1) * s_gx.norm(dim=-1))
            logs[f'{prefix}/{n}/cos(t_x,s_x).mean()'] = cossim.mean()

        dino.log_dict(logs)

    def on_train_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('train', outputs, dino)

    def on_validation_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('valid', outputs, dino)


class FeatureHistTracker(pl.Callback):       
    def step(self, prefix, out:Dict[str, torch.Tensor], dino:DINO):
        logs_wandb = {}
        prefix += '/feat'
        for n in ['embeddings', 'projections', 'logits']:
            t_x = out['teacher'][n].flatten(0,1)     # consider crops as batches
            s_x = out['student'][n].flatten(0,1)     # consider crops as batches
            s_gx = out['student'][n][:2].flatten(0,1)  # consider crops as batches
            n = n[:5]                       # shortname for plots
            
            for i, x in [('t', t_x), ('s', s_x)]:
                logs_wandb[f'{prefix}/{n}/{i}_x.mean().hist()'] = wandb_histogram(x.mean(dim=0), 64)
                logs_wandb[f'{prefix}/{n}/{i}_x.std().hist()'] = wandb_histogram(x.std(dim=0), 64)

                # within batch cosine similarity distance
                cossim = batch_cossim(x)
                cossim_triu = cossim[torch.triu_indices(*cossim.shape, offset=1).unbind()] # upper triangular values
                logs_wandb[f'{prefix}/{n}/{i}_x.corr().hist()'] = wandb_histogram(cossim_triu, 64)
                
                # within batch l2 distance
                l2dist = batch_l2dist(x)
                l2dist_triu = l2dist[torch.triu_indices(*l2dist.shape, offset=1).unbind()] # upper triangular values
                logs_wandb[f'{prefix}/{n}/{i}_x.pdist().hist()'] = wandb_histogram(l2dist_triu, 64)

            # between student and teacher
            l2dist = (t_x - s_gx).norm(dim=-1)
            logs_wandb[f'{prefix}/{n}/l2(t_x,s_x).hist()'] = wandb_histogram(l2dist, 64)

            rmse = (t_x - s_gx).square().mean(dim=-1).sqrt()
            logs_wandb[f'{prefix}/{n}/rmse(t_x,s_x).hist()'] = wandb_histogram(rmse, 64)

            dot = (t_x * s_gx).sum(-1)
            cossim = dot / (t_x.norm(dim=-1) * s_gx.norm(dim=-1))
            logs_wandb[f'{prefix}/{n}/cos(t_x,s_x).hist()'] = wandb_histogram(cossim, 64)

        #logs['trainer/global_step'] = dino.global_step
        #dino.logger.experiment.log(logs)
        dino.logger.log_metrics(logs_wandb, step=max(dino.global_step - 1, 0)) # -1 because global step is updated just before on train_batch_end

    def on_train_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('train', outputs, dino)

    def on_validation_batch_end(self, _: pl.Trainer, dino: DINO, outputs:Dict[str, torch.Tensor], *args) -> None:
        self.step('valid', outputs, dino)


class HParamTracker(pl.Callback):  
    def on_train_batch_start(self, _:pl.Trainer, dino:DINO, *args) -> None:
        logs, logs_wandb = {}, {}
        logs['hparams/lr'] = dino.optimizer.param_groups[0]['lr']
        logs['hparams/wd'] = dino.optimizer.param_groups[0]['weight_decay']
        logs['hparams/t_mom'] = dino.t_updater.mom
        logs['hparams/t_cmom'] = dino.teacher.head.cmom
        logs['hparams/s_cmom'] = dino.student.head.cmom
        logs['hparams/t_temp'] = dino.teacher.head.temp
        logs['hparams/s_temp'] = dino.student.head.temp
        logs['hparams/t_cent.norm()'] = dino.teacher.head.cent.norm()
        logs['hparams/t_cent.mean()'] = dino.teacher.head.cent.mean()
        logs_wandb['hparams/t_cent'] = wandb_histogram(dino.teacher.head.cent)
        
        dino.log_dict(logs)
        dino.logger.log_metrics(logs_wandb , step=dino.global_step)
        #logs['trainer/global_step'] = dino.global_step
        #dino.logger.experiment.log(logs)

class ParamTracker(pl.Callback):
    def __init__(self, student, teacher, name=None, track_init:bool=False) -> None:
        self.student:torch.nn.Module = student
        self.teacher:torch.nn.Module = teacher
        self.track_init = track_init
        self.name = '' if name is None else name + '/'
        self.g_vec = None
        
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

        logs[f'params/{self.name}nans(teach)'] = float(torch.isnan(t_vec).sum())
        logs[f'params/{self.name}nans(stud)'] = float(torch.isnan(s_vec).sum())
        logs[f'params/{self.name}infs(teach)'] = float(torch.isinf(t_vec).sum())
        logs[f'params/{self.name}infs(stud)'] = float(torch.isinf(s_vec).sum())

        # log position and angle relative to origin
        t_norm = torch.norm(t_vec)
        s_norm = torch.norm(s_vec)
        logs[f'params/{self.name}norm(teach)'] = t_norm
        logs[f'params/{self.name}norm(stud)'] = s_norm
        logs[f'params/{self.name}cos(teach, stud)'] = torch.dot(t_vec, s_vec) / (t_norm * s_norm)

        logs[f'params/{self.name}rms(teach)'] = t_vec.square().mean().sqrt()
        logs[f'params/{self.name}rms(stud)'] = s_vec.square().mean().sqrt()

        # get driving signals: gradient and difference
        g_vec = self.g_vec #U.module_to_vector(self.student, grad=True)
        d_vec = t_vec - s_vec

        # log driving signals
        g_norm = torch.norm(g_vec)
        d_norm = torch.norm(d_vec)
        logs[f'params/{self.name}norm(grad)'] = g_norm
        logs[f'params/{self.name}norm(diff)'] = d_norm
        logs[f'params/{self.name}cos(grad, diff)'] = torch.dot(g_vec, d_vec) / (g_norm * d_norm)

        logs[f'params/{self.name}rms(grad)'] = g_vec.square().mean().sqrt()
        logs[f'params/{self.name}rms(diff)'] = d_vec.square().mean().sqrt()

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
            
            logs[f'params/{self.name}rms(teach - init)'] = t_vec.square().mean().sqrt()
            logs[f'params/{self.name}rms(stud - init)'] = s_vec.square().mean().sqrt()

        self.g_vec = None
        dino.log_dict(logs)


class GradVarTracker(pl.Callback):
    def __init__(self, model:torch.nn.Module, submodels:dict={}) -> None:
        super().__init__()
        self.model = model
        self.grad:Dict[str, torch.Tensor] = dict()

        self.submodels = {'':model} 
        for name, submodel in submodels.items():
            if name == '': # overwrite behaviour of ''
                self.submodels[''] = submodel
            else: # rename to add '/'
                self.submodels[f'{name}/'] = submodel

    def on_after_backward(self, trainer: pl.Trainer, dino: DINO) -> None:
        for name, submodel in self.submodels.items():
            self.grad[name] = U.module_to_vector(submodel, grad=True)

    def on_train_batch_end(self, trainer: pl.Trainer, dino: DINO, outputs, batch: Any, batch_idx: int, *args) -> None:
        batch, batch_targets = batch

        # freeze batch statistics
        dino.train(False)

        self.model.zero_grad(True)
        variance = dict({(name, 0.0) for name in self.submodels.keys()})
        for i in range(len(batch_targets)):
            # prepare sample by performing one element slice
            sample = [crop[i:i+1] for crop in batch], batch_targets[i:i+1]

            # compute gradient in validation mode (without other updates)
            with torch.enable_grad():
                out = dino.validation_step(sample, batch_idx)
            out['loss'].backward()

            # accumulate sample gradients
            for name, submodel in self.submodels.items():
                sample_grad = U.module_to_vector(submodel, grad=True)
                variance[name] += torch.square(self.grad[name] - sample_grad).sum()          
            self.model.zero_grad(True)
        self.grad = dict()
        
        logs = dict()
        for name in self.submodels.keys():
            logs[f'params/{name}var(grad)'] = variance[name] / (len(batch_targets) - 1)
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

        # save features of before training
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

    @staticmethod
    def load_data(dir:str, start:int=None, stop:int=None, step:int=None) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        from glob import glob
        from tqdm import tqdm

        if not os.path.isdir(dir):
            raise RuntimeError(f'Is not a directory: \'{dir}\'')

        # get filenames
        fnames = sorted(glob(os.path.join(dir, 'table_*.pckl')))
        if len(fnames) == 0:
            raise RuntimeError(f'No data in directory: \'{dir}\'')


        data, columns, mem = [], None, 0
        loading_pbar = tqdm(fnames[start:stop:step])
        for fname in loading_pbar:
            # load pandas dataframe, extract columnames only once
            df = pd.read_pickle(fname)
            columns = df.columns if columns is None else columns
            
            # load table into torch
            table = torch.from_numpy(df.values)
            data.append(table)

            # log memory of all data
            mem += table.element_size() * table.nelement()
            loading_pbar.set_postfix({'mem':f'{mem*1e-6:.1f}MB'})

        # gather data in one tensor frame
        data = torch.stack(data, dim=0)

        # add indices to data
        indices = torch.arange(*slice(start, stop, step).indices(len(fnames)))

        return data, list(columns), indices

class ParamStatSaver(pl.Callback):
    def __init__(self, model:torch.nn.Module, name:str, dir:str) -> None:
        super().__init__()
        self.model = model
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

        self.grad_fname = os.path.join(self.dir, f'{name}_grad_stats.pckl')
        self.param_fname = os.path.join(self.dir, f'{name}_param_stats.pckl')
        
        self.grad_stats = [] 
        self.param_stats = []

    def on_after_backward(self, trainer: pl.Trainer, dino: DINO) -> None:
        stats = dict((n, p.grad.square().mean().sqrt().item()) for n, p in self.model.named_parameters())
        self.grad_stats.append(stats)

    def on_train_batch_end(self, trainer:pl.Trainer, dino: DINO, *args) -> None:
        stats = dict((n, p.square().mean().sqrt().item()) for n, p in self.model.named_parameters())
        self.param_stats.append(stats)

    def on_train_epoch_end(self, *args) -> None:
        pd.to_pickle(pd.DataFrame(self.grad_stats), self.grad_fname)
        pd.to_pickle(pd.DataFrame(self.param_stats), self.param_fname)