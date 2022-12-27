from math import sqrt
from time import time
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

__all__ = [
    'LinearProbe',
    'KNNProbe'
    'Prober'
]

class Probe():
    id = ''
    def prepare(self, n_features:int, n_classes:int, device:torch.device=None, generator:torch.Generator=None) -> None:   
        raise NotImplementedError()

    def train(self, train_data:List[Tuple[torch.Tensor]]) -> None:
        raise NotImplementedError()

    def valid(self, valid_data:List[Tuple[torch.Tensor]]) -> float:
        raise NotImplementedError()
    
    def cleanup(self) -> None:
        raise NotImplementedError()


class LinearProbe(Probe):
    id = 'lin'
    def __init__(self, n_epochs:int):
        super().__init__()
        self.n_epochs = n_epochs
        self.clf = None
        self.opt = None
        self.acc = None

    def prepare(self, n_features:int, n_classes:int, device:torch.device=None, generator:torch.Generator=None):   
        from .modules import init

        self.clf = nn.Linear(n_features, n_classes)
        self.opt = AdamW(self.clf.parameters())
        self.acc = Accuracy()
        if device and device.type == 'cuda':
            self.clf = self.clf.to(device=device)
            self.acc = self.acc.to(device=device)

        #m.reset_parameters() is equal to:
        bound = 1 / sqrt(self.clf.in_features)
        init.uniform_(self.clf.weight, -bound, bound, generator=generator)
        if self.clf.bias is not None:
            init.uniform_(self.clf.bias, -bound, bound, generator=generator)

    def train(self, train_data:List[Tuple[torch.Tensor]]):
        self.clf.train()
        train_pbar = tqdm(range(self.n_epochs), leave=False)
        train_pbar.set_description(f'Training')

        for epoch in train_pbar: # training
            for embeddings, targets in train_data: 
                self.opt.zero_grad(set_to_none=True) # step clf parameters

                with torch.enable_grad():
                    loss = F.cross_entropy(self.clf(embeddings), targets)
                loss.backward()
                self.opt.step()

            train_pbar.set_postfix({'loss':float(loss)})

    def valid(self, valid_data:List[Tuple[torch.Tensor]]):
        self.clf.eval()
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description('Validation')

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            predictions = self.clf(embeddings)
            self.acc.update(predictions, targets)
            valid_pbar.set_postfix({'acc':float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        self.clf = None
        self.opt = None
        self.acc = None

        
class KNNProbe(Probe):
    id = 'knn'
    def __init__(self, k:int):
        super().__init__()
        self.k = k
        self.index = None
        self.labels = None
        self.acc = None

    def prepare(self, n_features:int, n_classes:int=0, device:torch.device=None, generator:torch.Generator=None):   
        import faiss
        import faiss.contrib.torch_utils

        self.index = faiss.IndexFlat(n_features)
        self.acc = Accuracy()
        if device and device.type == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, device.index, self.index)
            self.acc = self.acc.to(device=device)

    def train(self, train_data:List[Tuple[torch.Tensor]]):
        train_pbar = tqdm(train_data, leave=False)
        train_pbar.set_description(f'Training')

        labels = []
        for embeddings, targets in train_pbar: 
            self.index.add(embeddings)
            labels.append(targets)
        self.labels = torch.cat(labels)

    def valid(self, valid_data:List[Tuple[torch.Tensor]]):
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description('Validation')

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            _, indices = self.index.search(embeddings, self.k)
            predictions = self.labels[indices].mode()[0]

            self.acc.update(predictions, targets)
            valid_pbar.set_postfix({'acc':float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        self.index = None
        self.labels = None
        self.acc = None


class Prober(pl.Callback):
    def __init__(self,
            encoders: Dict[str, nn.Module],
            probes: Dict[str, Probe],
            n_classes: int,
            train_dl:DataLoader, 
            valid_dl:DataLoader,
            probe_every:int = 1,
            seed = None
            ):
        super().__init__()

        self.probe_every = probe_every
        self.encoders = encoders
        self.probes = probes

        self.n_classes = n_classes
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.seed = seed

    def load_data(self, encoder:nn.Module, dl:DataLoader, device:torch.device=None) -> List[torch.Tensor]:
        loading_pbar = tqdm(dl, leave=False)
        loading_pbar.set_description(f'Loading embeddings')

        # store training mode and switch to eval
        mode = encoder.training
        encoder.eval()

        train_data, mem = [], 0
        with torch.no_grad():
            for batch in loading_pbar:
                inputs, targets = batch[0], batch[1]
                if device:
                    inputs, targets = inputs.to(device), targets.to(device)
                
                embeddings:torch.Tensor = encoder(inputs)
                train_data.append((embeddings, targets))
                
                mem += embeddings.element_size() * embeddings.nelement()
                loading_pbar.set_postfix({'mem':f'{mem*1e-6:.1f}MB'})

        # restore previous mode
        encoder.train(mode)
        return train_data

    @torch.no_grad()
    def probe(self, device=None):

        out = {}
        for enc_id, encoder in self.encoders.items():
            print(f'\nStarting probes {list(self.probes.keys())} of {enc_id}..', end='')
            t = time() 

            # prepare data: training data is random if seed is None and shuffle=True
            if self.train_dl.generator is not None:
                if self.seed is None:
                    self.train_dl.generator.seed()
                else:
                    self.train_dl.generator.manual_seed(self.seed) 

            # load data
            train_data = self.load_data(encoder, self.train_dl, device=device)
            valid_data = self.load_data(encoder, self.valid_dl, device=device)
            n_features = train_data[0][0].shape[-1]

            out[enc_id] = {}
            for probe_id, probe in self.probes.items():
  
                # prepare probe: everything is random if seed is None
                generator = torch.Generator(device=device)
                if self.seed is None:
                    generator.seed()
                else:
                    generator.manual_seed(self.seed)

                probe.prepare(n_features, self.n_classes, device, generator)
                
                probe.train(train_data)
                out[enc_id][probe_id] = probe.valid(valid_data)            

                probe.cleanup() # free space

            t = time() - t
            m, s = int(t//60), int(t%60)
            accs = [f'{acc:.3}' for acc in out[enc_id].values()]
            print(f' ..{enc_id} took {m:02d}:{s:02d}min \t=> accs = {accs}', end='')
        
        print('', end='\n')
        return out
    
    def probe_to_log(self, out):
        log = {}
        for enc_id, probes in out.items():
            for probe_id, acc in probes.items():
                id = f'probe/{enc_id}'
                if probe_id != '':
                    id += f'/{probe_id}'
                log[id] = acc
        return log

    # trainer.validate() needs to be called before trainer.fit() for per-training probe
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.probe_every == 0: # only probe every so many epochs
            pl_module.log_dict(self.probe_to_log(self.probe(pl_module.device)))
        
        elif trainer.current_epoch == trainer.max_epochs - 1: # probe after last epoch
            pl_module.log_dict(self.probe_to_log(self.probe(pl_module.device)))


if __name__ == '__main__':
    import argparse
    from random import shuffle

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()

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

    # prepare dataloaders
    train_dl = DataLoader(dataset=train_set, shuffle=True, batch_size=10)
    valid_dl = DataLoader(dataset=valid_set)
    
    # prepare prober
    prober = Prober(encoders = {'':nn.Identity()}, 
                    probes = {'lin': LinearProbe(n_epochs=100),
                                'knn': KNNProbe(k=20)},
                    train_dl=train_dl,
                    valid_dl=valid_dl,
                    n_classes=2)

    # train and validate
    prober.probe()