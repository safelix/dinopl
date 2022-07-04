from time import time
from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from tqdm import tqdm

import my_utils as U

class LinearProber(pl.Callback):
    def __init__(self,
            encoders:Dict[str, nn.Module],
            embed_dim:int,
            n_classes:int,
            probe_every:int, 
            probing_epochs:int,
            train_set:Dataset, 
            valid_set:Dataset,
            dl_args:Dict,
            ):
        super().__init__()

        self.probe_every = probe_every
        self.probing_epochs = probing_epochs

        self.encoders = encoders
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        self.train_set =  train_set
        self.valid_set =  valid_set

        self.dl_args = dl_args
        self.train_dl = None
        self.valid_dl = None


    def on_fit_start(self, *_) -> None:
        self.train_dl = DataLoader(dataset=self.train_set, **self.dl_args)
        self.valid_dl = DataLoader(dataset=self.valid_set, **self.dl_args)
 
    @torch.enable_grad()
    def probe(self, device):
        out = {}
        for name, encoder in self.encoders.items():
            accuracy = Accuracy().to(device=device)
            clf = nn.Linear(self.embed_dim, self.n_classes, device=device)
            opt = torch.optim.AdamW(clf.parameters())

            t = time()

            loading_pbar = tqdm(self.train_dl, leave=False)
            loading_pbar.set_description(f'Loading {name} embeddings')
            train_embs, mem = [], 0
            with torch.no_grad():
                for batch in loading_pbar:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    embeddings = encoder(inputs)
                    train_embs.append((embeddings, targets))
                    mem += embeddings.element_size() * embeddings.nelement()
                    loading_pbar.set_postfix({'mem':f'{mem*1e-6:.1f}MB'})      

            print('', flush=True, end='')
            train_pbar = tqdm(range(self.probing_epochs), leave=False)
            train_pbar.set_description(f'Training')
            for epoch in train_pbar: # training
                for embeddings, targets in train_embs: 
                    opt.zero_grad(set_to_none=True) # step clf parameters
                    loss = F.cross_entropy(clf(embeddings), targets)
                    loss.backward()
                    opt.step()
                train_pbar.set_postfix({'loss':float(loss)})

            valid_pbar = tqdm(self.valid_dl, leave=False)
            valid_pbar.set_description('Validation')
            with torch.no_grad(): 
                for batch in valid_pbar:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    accuracy.update(clf(encoder(inputs)), targets)
                    valid_pbar.set_postfix({'acc':float(accuracy.compute())})
                out[name] = float(accuracy.compute())

            t = time() - t
            grad = U.module_to_vector(clf, grad=True).norm()
            print(f' ..{name} took {int(t//60):02d}:{int(t%60):02d}min \t\t=> acc={out[name]:.3f}, grad={grad:.3f}', end='')
        print('')
        return dict((f'probe/{k}', v) for (k,v) in out.items())
    
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        if trainer.current_epoch % self.probe_every == 0: # only probe every so many epochs
            pl_module.log_dict(self.probe(pl_module.device)) 
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        if trainer.current_epoch == trainer.max_epochs - 1: # probe after last epoch
            pl_module.log_dict(self.probe(pl_module.device)) 
