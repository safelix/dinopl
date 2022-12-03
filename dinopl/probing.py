from math import sqrt
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

from .modules import init

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

    def reset(self, dev, generator=None):   
        self.dev = dev     
        self.clf = nn.Linear(self.embed_dim, self.n_classes, device=dev)
        self.opt = AdamW(self.clf.parameters())
        self.acc = Accuracy().to(device=dev)

        #m.reset_parameters() is equal to:
        bound = 1 / sqrt(self.clf.in_features)
        init.uniform_(self.clf.weight, -bound, bound, generator=generator)
        if self.clf.bias is not None:
            init.uniform_(self.clf.bias, -bound, bound, generator=generator)

    def load_data(self, dl:DataLoader):
        loading_pbar = tqdm(dl, leave=False)
        loading_pbar.set_description(f'Loading embeddings')

        # store training mode and switch to eval
        mode = self.encoder.training
        self.encoder.eval()

        train_data, mem = [], 0
        with torch.no_grad():
            for batch in loading_pbar:
                inputs, targets = batch[0].to(self.dev), batch[1].to(self.dev)
                
                embeddings = self.encoder(inputs)
                train_data.append((embeddings, targets))
                
                mem += embeddings.element_size() * embeddings.nelement()
                loading_pbar.set_postfix({'mem':f'{mem*1e-6:.1f}MB'})

        # restore previous mode
        self.encoder.train(mode)
        return train_data

    @torch.enable_grad()
    def train(self, epochs, train_data:List[Tuple[torch.Tensor]]):
        self.clf.train()
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
        self.clf.eval()
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
            train_dl:DataLoader, 
            valid_dl:DataLoader,
            seed = None
            ):
        super().__init__()

        self.probe_every = probe_every
        self.probing_epochs = probing_epochs
        self.probes = probes

        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.seed = seed

    def probe(self, device):

        out = {}
        for id, probe in self.probes.items():
  
            # prepare dataloader: everything is random if seed is None
            generator = torch.Generator(device=device)
            if self.seed is not None:
                generator.manual_seed(self.seed)
                if self.train_dl.generator is not None:
                    self.train_dl.generator.manual_seed(self.seed)

            probe.reset(device, generator)

            print(f'\nStarting {type(probe).__name__} of {id}..', end='')

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
    
    # trainer.validate() needs to be called before trainer.fit() for per-training probe
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.probe_every == 0: # only probe every so many epochs
            pl_module.log_dict(self.probe(pl_module.device))
        
        elif trainer.current_epoch == trainer.max_epochs - 1: # probe after last epoch
            pl_module.log_dict(self.probe(pl_module.device))


if __name__ == '__main__':
    import argparse
    from random import shuffle

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()

    # make probe
    probe = LinearProbe(
        encoder=torch.nn.Identity(),
        embed_dim=2,
        n_classes=2)

    # generate clustered data
    xs = torch.cat([torch.normal(-1.0, 1.0, (args.n_samples // 2, 2)),
                        torch.normal(+1.0, 1.0, (args.n_samples // 2, 2))])
    lbls = torch.cat([torch.full((args.n_samples // 2,), 0),
                        torch.full((args.n_samples // 2,), 1)])

    # build data sets
    data_set = list(zip(xs, lbls))
    shuffle(data_set)
    train_set = data_set[:args.n_samples // 2]
    valid_set = data_set[args.n_samples // 2:]

    # load data
    probe.reset(torch.device('cpu'))
    train_dl = DataLoader(dataset=train_set, shuffle=True, batch_size=10)
    valid_dl = DataLoader(dataset=valid_set)
    train_data = probe.load_data(train_dl)
    valid_data = probe.load_data(valid_dl)

    # train and validate
    probe.train(100, train_data=train_data)
    acc = probe.valid(valid_data=valid_data)

    print(f'Accuracy: {acc:.3f}')