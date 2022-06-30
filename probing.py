from time import time
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm



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
            pl_module.log_dict(self.probe(pl_module.device)) 

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        if trainer.current_epoch % self.probe_every == 0: # only probe every so many epochs
            pl_module.log_dict(self.probe(pl_module.device)) 
        
        elif trainer.current_epoch == trainer.max_epochs - 1: # probe after last epoch
            pl_module.log_dict(self.probe(pl_module.device)) 
