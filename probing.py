from time import time
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

import my_utils as U


class LinearProbe(pl.LightningModule):
    def __init__(self, encoder:nn.Module, embed_dim:int, n_classes:int):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.encoder = encoder
        self.linear = nn.Linear(embed_dim, n_classes)

    def forward(self, batch):
        mode = self.encoder.training

        # TODO: inefficient? move outside of training loop?
        self.encoder.train(False) # same as .eval()
        with torch.no_grad():
            embs = self.encoder(batch)
        self.encoder.train(mode) # restore mode

        return self.linear(embs)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.linear.parameters()) 
    
    def training_step(self, batch, batch_idx):
        # TODO: why is this list of length 1, but not in validation_step()?!
        #raise Exception('\n'+U.recprint(batch))
        inputs, targets = batch[0]
        return F.cross_entropy(self(inputs), targets)
    
    def on_validation_epoch_start(self):
        self.accuracy = Accuracy()
        
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        self.accuracy.update(self(inputs), targets)

    def validation_epoch_end(self, outputs):
        return self.accuracy.compute()
    
        
class ProbingCallback(pl.Callback):
    def __init__(self, 
            probes:Dict[str, LinearProbe], 
            train_dl:DataLoader, 
            valid_dl:DataLoader, 
            probe_every:int, 
            probing_epochs:int):
        super().__init__()

        self.probes = probes
        self.train_dl =  train_dl
        self.valid_dl =  valid_dl

        self.probe_every = probe_every
        self.probing_epochs = probing_epochs
        self.probe_trainer = pl.Trainer(max_epochs=probing_epochs,
                                        accelerator='auto')

    def probe(self):
        acc = {}
        for name, probe in self.probes.items():

            print(f'Starting LinearProbe of {name}... ', flush=True, end='')

            t = time()
            self.probe_trainer.fit(model=probe, 
                                    train_dataloaders=[self.train_dl],
                                    val_dataloaders=[self.valid_dl])
            acc[name] = self.probe_trainer.validate(probe, dataloaders=[self.valid_dl])
            t = time() - t

            print(f'took {int(t//60):02d}:{int(t%60):02d}min \t\t=> acc: {acc[name]:.3f}', flush=True)
        
        return dict((f'probe_{k}', v) for (k,v) in acc.items())


    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        pl_module.log_dict(self.probe()) 

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        # only probe every so many epochs
        if trainer.current_epoch % self.probe_every == 0: 
            pl_module.log_dict(self.probe()) 
     
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
            pl_module.log_dict(self.probe()) 


        