from typing import Dict
import pytorch_lightning as pl
from torch import nn
import torch
from dino import DINO
from time import time, strftime
from torchmetrics import Accuracy

from torch.nn import functional as F
from configuration import Configuration, create_optimizer

class LinearProbe(pl.LightningModule):
    def __init__(self, config:Configuration, encoder:nn.Module):
        super().__init__()

        self.n_classes = config.n_classes
        self.encoder = encoder
        self.linear = nn.Linear(config.embed_dim, config.n_classes)

    def forward(self, batch):
        mode = self.encoder.training

        self.encoder.train(False) # same as .eval()
        with torch.no_grad():
            embs = self.encoder(batch)
        self.encoder.train(mode) # restore mode

        return self.linear(embs)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.linear) 
    
    def training_step(self, batch):
        inputs, targets = batch
        return F.cross_entropy(self(inputs), targets)
    
    def on_validation_epoch_start(self):
        self.accuracy = Accuracy()
        
    def validation_step(self, batch):
        inputs, targets = batch
        self.accuracy.update(self(inputs), targets)

    def validation_epoch_end(self):
        return self.accuracy.compute()
    
        
class ProbingCallback(pl.Callback):
    def __init__(self, config, probes:Dict[str, LinearProbe], train_dl, valid_dl):
        super().__init__()

        self.probes = probes
        self.train_dl =  train_dl
        self.valid_dl =  valid_dl

        self.probe_every = config.probe_every
        self.probe_trainer = pl.Trainer(max_epochs=config.probing_epochs,
                                        accelerator='auto')

    def probe(self):
        acc = {}
        for name, probe in self.probes:

            print(f'Starting LinearProbe of {name}... ', flush=True, end='')

            t = time()
            self.probe_trainer.fit(probe, self.train_dl)
            acc[name] = self.probe_trainer.validate(probe, self.valid_dl)
            t = time() - t

            print(f'took {int(t//60):02d}:{int(t%60):02d}min \t\t=> acc: {acc[name]:.3f}', flush=True)
        
        return acc

    def on_epoch_end(self, trainer: pl.Trainer):
        if trainer.current_epoch % self.probe_every == 0:
            trainer.log(self.probe()) # only probe every so many epochs
     
    def on_train_end(self):
        self.probe()
        

        