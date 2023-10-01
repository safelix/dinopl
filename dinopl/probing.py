from math import sqrt
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from warnings import warn

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

__all__ = [
    'LinearAnylsis',
    'KNNAnalysis'
    'LogRegAnalysis'
    'LinDiscrAnalysis'
    'Prober'
]

class Analysis(object):
    def prepare(self, n_features:int, n_classes:int, device:torch.device=None, generator:torch.Generator=None) -> None:   
        raise NotImplementedError()

    def train(self, train_data:List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        raise NotImplementedError()

    def valid(self, valid_data:List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        raise NotImplementedError()
    
    def cleanup(self) -> None:
        raise NotImplementedError()
    
    def __del__(self):
        self.cleanup()

class LinearAnalysis(Analysis):
    def __init__(self, n_epochs:int):
        super().__init__()
        self.n_epochs = n_epochs
        self.clf = None
        self.opt = None
        self.acc = None
        self.device = torch.device('cpu')

    @torch.inference_mode(False)
    @torch.no_grad() 
    def prepare(self, n_features:int, n_classes:int, device:torch.device=None, generator:torch.Generator=None):   
        from .modules import init

        self.clf = nn.Linear(n_features, n_classes)
        self.opt = AdamW(self.clf.parameters())
        self.acc = Accuracy(task='multiclass', num_classes=n_classes)
        if device and device.type == 'cuda':
            self.clf = self.clf.to(device=device)
            self.acc = self.acc.to(device=device)
            self.device = device

        #m.reset_parameters() is equal to:
        bound = 1 / sqrt(self.clf.in_features)
        init.uniform_(self.clf.weight, -bound, bound, generator=generator)
        if self.clf.bias is not None:
            init.uniform_(self.clf.bias, -bound, bound, generator=generator)

    @torch.inference_mode(False)
    @torch.no_grad()
    def train(self, train_data:List[Tuple[torch.Tensor, torch.Tensor]], verbose=True):
        self.clf.train()
        train_pbar = tqdm(range(self.n_epochs), leave=False)
        train_pbar.set_description(f'Training')

        for epoch in train_pbar: # training
            for embeddings, targets in train_data: 
                self.opt.zero_grad(set_to_none=True) # step clf parameters

                with torch.enable_grad():
                    loss = F.cross_entropy(self.clf(embeddings.to(self.device)), targets.to(self.device))
                loss.backward()
                self.opt.step()

            train_pbar.set_postfix({'loss':float(loss)})

    def valid(self, valid_data:List[Tuple[torch.Tensor, torch.Tensor]]):
        self.clf.eval()
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description('Validation')

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            predictions = self.clf(embeddings.to(self.device))
            self.acc.update(predictions, targets.to(self.device))
            valid_pbar.set_postfix({'acc':float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        self.clf = None
        self.opt = None
        self.acc = None
        self.device = torch.device('cpu')

        
class KNNAnalysis(Analysis):
    id = 'knn'
    def __init__(self, k:int):
        super().__init__()
        self.k = k
        self.index = None
        self.labels = None
        self.acc = None
        self.device = torch.device('cpu')

    def prepare(self, n_features:int, n_classes:int=0, device:torch.device=None, generator:torch.Generator=None):   
        import faiss
        import faiss.contrib.torch_utils

        self.index = faiss.IndexFlat(n_features)
        self.acc = Accuracy(task='multiclass', num_classes=n_classes)
        if device and device.type == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, device.index, self.index)
            self.acc = self.acc.to(device=device)
            self.device = device

    def train(self, train_data:List[Tuple[torch.Tensor, torch.Tensor]]):
        train_pbar = tqdm(train_data, leave=False)
        train_pbar.set_description(f'Training')

        labels = []
        for embeddings, targets in train_pbar: 
            self.index.add(embeddings.to(self.device))
            labels.append(targets.to(self.device))
        self.labels = torch.cat(labels)

    def valid(self, valid_data:List[Tuple[torch.Tensor, torch.Tensor]]):
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description('Validation')

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            _, indices = self.index.search(embeddings.to(self.device), self.k)
            predictions = self.labels[indices].mode()[0]

            try: # catch bug from torchmetric?
                self.acc.update(predictions, targets.to(self.device))
            except Exception as e:
                warn(f'Could not compute accuracy: {str(e)})') 
            valid_pbar.set_postfix({'acc':float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        if self.index is not None:
            self.index.reset()
        self.index = None
        self.labels = None
        self.acc = None
        self.device = torch.device('cpu')


class LogRegAnalysis(Analysis):
    def __init__(self):
        super().__init__()
        self.clf = None
        self.opt = None
        self.acc = None
        self.device = torch.device('cpu')

    @torch.inference_mode(False)
    @torch.no_grad() 
    def prepare(self, n_features:int, n_classes:int, device:torch.device=None, generator:torch.Generator=None):   
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(multi_class='multinomial')
        self.acc = Accuracy(task='multiclass', num_classes=n_classes)

    @torch.inference_mode(False)
    @torch.no_grad()
    def train(self, train_data:List[Tuple[torch.Tensor, torch.Tensor]], verbose=True):
        X, y = zip(*train_data)
        X, y = torch.cat(X).numpy(), torch.cat(y).numpy()
        self.clf.fit(X, y)

    def valid(self, valid_data:List[Tuple[torch.Tensor, torch.Tensor]]):
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description('Validation')

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            predictions = self.clf.predict_proba(embeddings.numpy())
            self.acc.update(torch.from_numpy(predictions), targets)
            valid_pbar.set_postfix({'acc':float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        self.clf = None
        self.acc = None
        self.device = torch.device('cpu')

# LinearDiscriminantAnalysis supported on GPU:
# https://scikit-learn.org/stable/modules/array_api.html#pytorch-support
class LinDiscrAnalysis(Analysis):
    from sklearn import config_context
    
    def __init__(self):
        super().__init__()
        self.clf = None
        self.opt = None
        self.acc = None
        self.device = torch.device('cpu')

    @torch.inference_mode(False)
    @torch.no_grad() 
    @config_context(array_api_dispatch=True)
    def prepare(self, n_features:int, n_classes:int, device:torch.device=None, generator:torch.Generator=None):   
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        self.clf = LDA()
        self.acc = Accuracy(task='multiclass', num_classes=n_classes)

        if device and device.type == 'cuda':
            self.acc = self.acc.to(device=device)
            self.device = device

    @torch.inference_mode(False)
    @torch.no_grad()
    @config_context(array_api_dispatch=True)
    def train(self, train_data:List[Tuple[torch.Tensor, torch.Tensor]], verbose=True):
        X, y = zip(*train_data)
        X, y = torch.cat(X), torch.cat(y)
        self.clf.fit(X.to(self.device), y.to(self.device))

    @config_context(array_api_dispatch=True)
    def valid(self, valid_data:List[Tuple[torch.Tensor, torch.Tensor]]):
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description('Validation')

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            predictions = self.clf.predict_proba(embeddings.to(self.device))
            self.acc.update(predictions, targets.to(self.device))
            valid_pbar.set_postfix({'acc':float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        self.clf = None
        self.acc = None
        self.device = torch.device('cpu')


@torch.inference_mode()
def load_data(encoder:nn.Module, dl:DataLoader, device:torch.device=None):
    loading_pbar = tqdm(dl, leave=False)
    loading_pbar.set_description(f'Loading embeddings')

    # store training mode and switch to eval
    mode = encoder.training
    encoder.eval()

    data, mem = [], 0
    for batch in loading_pbar:
        inputs, targets = batch[0], batch[1]
        if device:
            inputs, targets = inputs.to(device), targets.to(device)
        
        embeddings:torch.Tensor = encoder(inputs)
        data.append((embeddings.contiguous().squeeze().cpu(), targets.cpu()))
        
        mem += embeddings.element_size() * embeddings.nelement()
        loading_pbar.set_postfix({'mem':f'{mem*1e-6:.1f}MB'})

    # restore previous mode
    encoder.train(mode)
    return data

@torch.inference_mode()
def normalize_data(train_data:List[Tuple[torch.Tensor, torch.Tensor]], 
                    valid_data:List[Tuple[torch.Tensor, torch.Tensor]]):

    # easier but with memory overhead to store entire dataset
    #embeddings = torch.cat(next(zip(*train_data)))
    #torch_mean = embeddings.mean(dim=0)
    #torch_std = embeddings.std(dim=0)

    n_features = train_data[0][0].shape[-1]
    n_samples = sum([emb.numel() for emb, _ in train_data]) / n_features
    mean = torch.stack([emb.sum(0) for emb, _ in train_data]).sum(0) / n_samples
    norm2 = torch.stack([(emb - mean).square().sum(0) for emb, _ in train_data]).sum(0) 
    std = torch.sqrt(norm2 / (n_samples - 1))

    # fill 0 like in sklearn.StandardScaler
    # https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/preprocessing/_data.py#L114
    std[std < 10 * torch.finfo(std.dtype).eps] = 1 

    [emb.sub_(mean).div_(std) for emb, _ in train_data]
    [emb.sub_(mean).div_(std) for emb, _ in valid_data]


class Prober(pl.Callback):
    def __init__(self,
            encoders: Dict[str, nn.Module],
            analyses: Dict[str, Analysis],
            train_dl:DataLoader, 
            valid_dl:DataLoader,
            n_classes: int,
            normalize:bool = False,
            probe_every:int = 1,
            seed = None
            ):
        super().__init__()

        self.encoders = encoders
        self.analyses = analyses

        self.n_classes = n_classes
        self.train_dl = train_dl
        self.valid_dl = valid_dl

        self.normalize = normalize
        self.probe_every = probe_every
        self.seed = seed 
    
    @torch.inference_mode()
    def eval_probe(self, train_data:List[Tuple[torch.Tensor, torch.Tensor]], 
                            valid_data:List[Tuple[torch.Tensor, torch.Tensor]], device=None):

        accs = {}
        n_features = train_data[0][0].shape[-1]
        for analysis_id, analysis in self.analyses.items():
  
            # prepare probe: everything is random if seed is None
            generator = torch.Generator(device=device)
            if self.seed is None:
                generator.seed()
            else:
                generator.manual_seed(self.seed)

            analysis.prepare(n_features, self.n_classes, device, generator)
            
            analysis.train(train_data)
            accs[analysis_id] = analysis.valid(valid_data)            

            analysis.cleanup() # free space

        return accs

    @torch.inference_mode()
    def probe(self, device_enc=None, device_emb=None, verbose=True):
        '''Args:
            - device_enc: device to encode images, should match encoder (None defaults to cpu)
            - device_emb: device to analyze embeddings, batches are used if possible (None defaults to device_enc)
        '''
        device_emb = device_emb or device_enc # use encoder device for embeddings by default

        out = {}
        for enc_id, encoder in self.encoders.items():
            if verbose:
                tqdm.write(f'\nStarting analyses {list(self.analyses.keys())} of {enc_id}..', end='')
                t = time() 

            # prepare data: training data is random if seed is None and shuffle=True
            if self.train_dl.generator is not None:
                if self.seed is None:
                    self.train_dl.generator.seed()
                else:
                    self.train_dl.generator.manual_seed(self.seed) 

            # load data
            train_data = load_data(encoder, self.train_dl, device=device_enc) # store embeddings on cpu
            valid_data = load_data(encoder, self.valid_dl, device=device_enc) # store embeddings on cpu

            # evaluate data
            accs = self.eval_probe(train_data, valid_data, device=device_emb) # move to device_emb on demand
            for key, val in accs.items():
                key = f'probe/{enc_id}' if key=='' else f'probe/{enc_id}/{key}'
                out[key] = val
            
            # evaluate normalized data
            if self.normalize:
                normalize_data(train_data, valid_data)
                accs = self.eval_probe(train_data, valid_data, device=device_emb) # move to device_emb on demand
                for key, val in accs.items():
                    key = f'probe/norm/{enc_id}' if key=='' else f'probe/norm/{enc_id}/{key}'
                    out[key] = val
    
            if verbose:
                t = time() - t
                tqdm.write(f' ..{enc_id} took {int(t//60):02d}:{int(t%60):02d}min', end='')

            # cleanup data
            del train_data, valid_data

        if verbose:
            tqdm.write(' => ' + str({key: f'{val:.3}' for key, val in out.items()}))

        return out 

    # trainer.validate() needs to be called before trainer.fit() for per-training probe
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        #if trainer.current_epoch % self.probe_every == 0: # only probe every so many epochs
        pl_module.log_dict(self.probe(pl_module.device))
        
        #elif trainer.current_epoch == trainer.max_epochs - 1: # probe after last epoch
        #    pl_module.log_dict(self.probe(pl_module.device))


from torchvision.datasets import VisionDataset
class ToySet(VisionDataset):
        img_size = (1, 1)
        ds_pixels = 1
        ds_channels = 2
        ds_classes = 2
        cmean, cstd = 2.0, 1.0 # cluster mean and center
        mean = torch.Tensor((0, 0))
        std = torch.Tensor((sqrt(cmean**2 + cstd**2), sqrt(cmean**2 + cstd**2)))
        def __init__(self, train: bool = True, n_samples = 100, **kwargs) -> None:
            self.n_samples = n_samples

            # generate clustered data
            self.data = torch.cat([torch.normal(-self.cmean, self.cstd, (n_samples // 2, 2)),
                                    torch.normal(+self.cmean, self.cstd, (n_samples // 2, 2))])

            # generate cluster labels
            self.lbls = torch.cat([torch.full((n_samples // 2,), 0),
                                    torch.full((n_samples // 2,), 1)])
            

            super().__init__(train, **kwargs)
        
        def __getitem__(self, index: int) -> Any:
            return (self.data[index] - self.mean) / self.std, self.lbls[index]
        
        def __len__(self) -> int:
            return self.n_samples


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples', type=int, default=100)
    parser.add_argument('--valid_samples', type=int, default=100)
    args = parser.parse_args()

    train_set = ToySet(train=True, n_samples=args.train_samples)
    valid_set = ToySet(train=False, n_samples=args.valid_samples)

    # prepare dataloaders
    train_dl = DataLoader(dataset=train_set, shuffle=True, batch_size=10)
    valid_dl = DataLoader(dataset=valid_set)
    
    # prepare prober
    prober = Prober(encoders = {'':nn.Identity()}, 
                    analyses = {'lin': LinearAnalysis(n_epochs=100),
                                'knn': KNNAnalysis(k=20)},
                    train_dl=train_dl,
                    valid_dl=valid_dl,
                    n_classes=2)

    # train and validate
    prober.probe()