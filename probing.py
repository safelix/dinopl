from time import time
from typing import Dict, List
from importlib_metadata import requires

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from tqdm import tqdm

import my_utils as U

class LinearProbe(pl.LightningModule):
    def __init__(self, encoder:nn.Module, embed_dim:int, n_classes:int):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.encoder = encoder
        self.linear = nn.Linear(embed_dim, n_classes)
        self.accuracy:Accuracy = None

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
    
    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        # TODO: why is this list of length 1, but not in validation_step()?!
        #raise Exception('\n'+U.recprint(batch))
        inputs, targets = batch[0]
        predictions = self(inputs)

        out = {'loss': F.cross_entropy(predictions, targets),
                'acc': accuracy(predictions, targets)}
        self.log_dict(out, prog_bar=True)
        return out
    
    def on_validation_epoch_start(self):
        self.accuracy = Accuracy().to(self.device)
        
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)

        # update state of aggregated accuracy
        self.accuracy.update(self(inputs), targets)

        # batchwise metrics
        out = {'val_loss': F.cross_entropy(predictions, targets),
                'val_acc': accuracy(predictions, targets)}
        self.log_dict(out, prog_bar=True)
        return out
    
        
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

    def probe(self, device):
        acc = {}
        for name, probe in self.probes.items():
            print(f'Starting LinearProbe of {name}... ', flush=True)
            t = time()

            # It seems this doesn't work on a single GPU?
            self.probe_trainer.fit(model=probe, 
                                    train_dataloaders=[self.train_dl],
                                    val_dataloaders=[self.valid_dl])
            self.probe_trainer.validate(probe, dataloaders=[self.valid_dl], verbose=False)
            acc[name] = float(probe.accuracy.compute())

            t = time() - t
            print(f'..took {int(t//60):02d}:{int(t%60):02d}min \t\t=> acc: {acc[name]:.3f}', flush=True)
            
        return dict((f'probe_{k}', v) for (k,v) in acc.items())

    #def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
    #    pl_module.log_dict(self.probe(), prog_bar=True) 

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        # only probe every so many epochs
        if trainer.current_epoch % self.probe_every == 0: 
            pl_module.log_dict(self.probe(pl_module.device), prog_bar=True) 
     
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
            pl_module.log_dict(self.probe(pl_module.device), prog_bar=True) 



###############################################################################
############################ Try to keep it simple ############################
###############################################################################

class LinearProbingCallback(pl.Callback):
    def __init__(self,
            encoders:Dict[str, nn.Module],
            embed_dim:int,
            n_classes:int,
            train_dl:DataLoader, 
            valid_dl:DataLoader, 
            probe_every:int, 
            probing_epochs:int):
        super().__init__()

        self.encoders = encoders
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        self.train_dl =  train_dl
        self.valid_dl =  valid_dl

        self.probe_every = probe_every
        self.probing_epochs = probing_epochs

    @torch.enable_grad()
    def probe(self, device):
        out = {}
        for name, encoder in self.encoders.items():
            accuracy = Accuracy().to(device=device)
            clf = nn.Linear(self.embed_dim, self.n_classes, device=device)
            opt = torch.optim.AdamW(clf.parameters())

            print(f'Starting LinearProbe of {name}... ', flush=True, end='')
            t = time()

            train_pbar = tqdm(range(self.probing_epochs), leave=False)
            train_pbar.set_description(f'Training')
            for epoch in train_pbar: # training
                epoch_pbar = tqdm(self.train_dl, leave=False)
                epoch_pbar.set_description(f'Epoch {epoch}')
                for batch in epoch_pbar: 
                    inputs, targets = batch[0].to(device), batch[1].to(device)

                    with torch.no_grad(): # get embeddings
                        embeddings = encoder(inputs)
                    opt.zero_grad(set_to_none=True) # step clf parameters
                    loss = F.cross_entropy(clf(embeddings), targets)
                    loss.backward()
                    opt.step()
                    epoch_pbar.set_postfix({'loss':float(loss)})      

            with torch.no_grad(): 
                valid_pbar = tqdm(self.valid_dl, leave=False)
                valid_pbar.set_description('Validation')
                for batch in valid_pbar:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    accuracy.update(clf(encoder(inputs)), targets)
                    valid_pbar.set_postfix({'acc':float(accuracy.compute())})
                out[name] = float(accuracy.compute())

            t = time() - t
            print(f'...took {int(t//60):02d}:{int(t%60):02d}min \t\t=> acc: {out[name]:.3f}', flush=True)
        return dict((f'{k}_acc', v) for (k,v) in out.items())
    

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
            pl_module.log_dict(self.probe(pl_module.device), prog_bar=True) 

    def on_training_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        if trainer.current_epoch % self.probe_every == 0: # only probe every so many epochs      
            pl_module.log_dict(self.probe(pl_module.device), prog_bar=True) 
     
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        if trainer.current_epoch % self.probe_every != 0: # only probe if on_training_epoch_end didn't probe yet      
            pl_module.log_dict(self.probe(pl_module.device), prog_bar=True) 
