from time import time
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from tqdm import tqdm

__all__ = [
    'LinearProbe', 
    'LinearProber'
]

class LinearProbe():
    def __init__(self, 
            encoder:torch.nn.Module,
            embed_dim:int,
            n_classes:int
            ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        self.dev = None
        self.clf = None
        self.opt = None

    def reset(self, dev):   
        self.dev = dev         
        self.clf = nn.Linear(self.embed_dim, self.n_classes, device=dev)
        self.opt = AdamW(self.clf.parameters())
        self.acc = Accuracy().to(device=dev)

    def load_data(self, dl:DataLoader):
        loading_pbar = tqdm(dl, leave=False)
        loading_pbar.set_description(f'Loading embeddings')

        train_data, mem = [], 0
        with torch.no_grad():
            for batch in loading_pbar:
                inputs, targets = batch[0].to(self.dev), batch[1].to(self.dev)
                
                embeddings = self.encoder(inputs)
                train_data.append((embeddings, targets))
                
                mem += embeddings.element_size() * embeddings.nelement()
                loading_pbar.set_postfix({'mem':f'{mem*1e-6:.1f}MB'})

        return train_data

    @torch.enable_grad()
    def train(self, epochs, train_data:List[Tuple[torch.Tensor]]):
        train_pbar = tqdm(range(epochs), leave=False)
        train_pbar.set_description(f'Training')

        for epoch in train_pbar: # training
            for embeddings, targets in train_data: 
                self.opt.zero_grad(set_to_none=True) # step clf parameters

                loss = F.cross_entropy(self.clf(embeddings), targets)
                loss.backward()
                self.opt.step()

            train_pbar.set_postfix({'loss':float(loss)})

    def valid(self, valid_data:List[Tuple[torch.Tensor]]):
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description('Validation')

        for embeddings, targets in valid_pbar:
            self.acc.update(self.clf(embeddings), targets)
            valid_pbar.set_postfix({'acc':float(self.acc.compute())})

        return float(self.acc.compute())
        


class LinearProber(pl.Callback):
    def __init__(self,
            probe_every:int, 
            probing_epochs:int,
            probes:Dict[str, LinearProbe],
            train_set:Dataset, 
            valid_set:Dataset,
            dl_args:Dict,
            ):
        super().__init__()

        self.probe_every = probe_every
        self.probing_epochs = probing_epochs
        self.probes = probes

        self.train_set =  train_set
        self.valid_set =  valid_set

        self.dl_args = dl_args
        self.train_dl = None
        self.valid_dl = None

    def on_fit_start(self, *_) -> None:
        self.train_dl = DataLoader(dataset=self.train_set, **self.dl_args)
        self.valid_dl = DataLoader(dataset=self.valid_set, **self.dl_args)

    def probe(self, device):
        out = {}
        for id, probe in self.probes.items():
            probe.reset(device)

            # load data
            t = time() 
            train_data = probe.load_data(self.train_dl)
            valid_data = probe.load_data(self.valid_dl)
            print('', flush=True, end='')

            # train and validate
            probe.train(self.probing_epochs, train_data)
            out[id] = probe.valid(valid_data)            

            t = time() - t
            m, s = int(t//60), int(t%60)
            print(f' ..{id} took {m:02d}:{s:02d}min \t\t=> acc={out[id]:.3f}', end='')
        print('')
        return dict((f'probe/{k}', v) for (k,v) in out.items())
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        if trainer.current_epoch % self.probe_every == 0: # only probe every so many epochs
            pl_module.log_dict(self.probe(pl_module.device)) 
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        if trainer.current_epoch == trainer.max_epochs - 1: # probe after last epoch
            pl_module.log_dict(self.probe(pl_module.device)) 
