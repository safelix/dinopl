from typing import List

import pytorch_lightning as pl
import torch

import my_utils as U
from dino import DINO


class HParamLogger(pl.Callback):
    def __init__(self, hparams:List[str]=None) -> None:
        self.hparams = hparams
    
    def log(self, dino:DINO):
        out = {
            'hparams/lr': dino.optimizer.param_groups[0]['lr'],
            'hparams/wd': dino.optimizer.param_groups[1]['weight_decay'],
            'hparams/t_mom': dino.t_updater.mom,
            'hparams/t_cmom': dino.teacher.head.cmom,
            'hparams/t_temp': dino.teacher.head.temp,
            'hparams/s_temp': dino.student.head.temp
        }

        if not None:
            out = dict((k, v) for (k,v) in out.items() if k in self.hparams)
        self.log_dir(out)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        self.log(pl_module)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        self.log(pl_module)


class ParamLogger(pl.Callback):
    def log(self, dino:DINO):
        i_vec = U.module_to_vector(dino.init)
        s_vec = U.module_to_vector(dino.student)
        t_vec = U.module_to_vector(self.teacher)

        self.log_dict({
            'params/dot(i,t)' : torch.dot(i_vec, t_vec),
            'params/dot(i,s)' : torch.dot(i_vec, s_vec),
            'params/dot(t,s)' : torch.dot(t_vec, s_vec),
            'params/norm(i-t)' : torch.norm(i_vec - t_vec),
            'params/norm(i-s)' : torch.norm(i_vec - s_vec),
            'params/norm(t-s)' : torch.norm(t_vec - s_vec)
        })

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        self.log(pl_module)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        self.log(pl_module)
