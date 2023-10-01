

import os, gc
import pickle
import argparse
from glob import glob
from warnings import warn
from math import sqrt, floor, ceil

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from typing import Dict, List, Tuple

import wandb
from tqdm import tqdm
from functools import reduce
from configuration import load_data, load_model
from dinopl.utils import pick_single_gpu
api = wandb.Api(timeout=19)

from torch.utils.data import DataLoader

class ModelHook():
    def __init__(self, model:nn.Module, rootname='', layers='root') -> None:
        if layers not in ['root', 'leafs']:
            raise ValueError('Layer must be either \'root\' or \'leafs\'.')

        self.names : Dict[nn.Module, str] = {}
        self.fwd_handles : Dict[str, RemovableHandle] = {}
        self.bwd_handles : Dict[str, RemovableHandle] = {}
        
        root:nn.Module = model
        if rootname != '': # get root model
            root = reduce(getattr, rootname.split('.'), model)
        
        if layers == 'root': # only add root and return
            self.names[root] = rootname
            return

        # else layers == 'leafs'
        if rootname != '': # prepare rootname for concatenation
            rootname += '.'
        for subname, submodule in root.named_modules():
            if sum(1 for _ in submodule.children()) == 0:
                self.names[submodule] = rootname+subname
        return

    def register_hooks(self):
        for module, name in self.names.items():
            self.fwd_handles[name] = module.register_forward_hook(self.forward_hook)
            self.bwd_handles[name] = module.register_backward_hook(self.backward_hook)

    def remove_hooks(self):
        for name, handle in self.fwd_handles.items():
            handle.remove()
        self.fwd_handles = {}
        for name, handle in self.bwd_handles.items():
            handle.remove()
        self.bwd_handles = {}

    def __del__(self) -> None:
        self.remove_hooks()

    def forward_hook(self, module, input, output) -> None:
        pass

    def backward_hook(self, module, grad_input, grad_output) -> torch.Tensor:
        pass


class EmbeddingLoader(ModelHook):
    def __init__(self, model:nn.Module, rootname='', layers='root') -> None:
        super().__init__(model, rootname, layers)
        self.embeddings:Dict[str, torch.Tensor] = {}
        self.targets:torch.Tensor = []
        
        self.model = model
        if rootname != '':
            self.model = reduce(getattr, rootname.split('.'), model)
        self.embeddings = {n:[] for n in self.names.values()}

    def forward_hook(self, module, input, output) -> None:
        self.embeddings[self.names[module]].append(output.detach().cpu())

    @torch.no_grad()
    @torch.inference_mode()
    def load_data(self, dl:DataLoader, device=torch.device('cpu')):
        self.register_hooks()
        self.embeddings = {n:[] for n in self.names.values()}
        self.targets = []

        for inputs, targets in tqdm(dl, desc='Loading Data'):
            self.targets.append(targets)
            _ = self.model(inputs.to(device))

        self.targets = torch.cat(self.targets, dim=0)
        for name in tqdm(self.embeddings.keys(), 'Merging Data'):
            if isinstance(self.embeddings[name], torch.Tensor):
                continue
            self.embeddings[name] = torch.cat(tuple(b.flatten(1) for b in self.embeddings[name]), dim=0) 
        self.remove_hooks()

    @torch.no_grad()
    @torch.inference_mode()
    def center(self, mean={}, device=torch.device('cpu')):
        for k in tqdm(self.embeddings.keys(), desc='Centering Data'):
            data:torch.Tensor = self.embeddings[k].to(device)
            if k not in mean.keys():
                mean[k] = data.mean(dim=0)
            self.embeddings[k] = (data - mean[k]).cpu()
        return dict(mean=mean)
    
    @torch.no_grad()
    @torch.inference_mode()
    def standardize(self, mean={}, std={}, device=torch.device('cpu')):
        mean, std = {}, {}
        for k in tqdm(self.embeddings.keys(), desc='Standardizing Data'):
            data:torch.Tensor = self.embeddings[k].to(device)
            if k not in mean.keys():
                mean[k] = data.mean(dim=0)
            if k not in std.keys():
                std[k] = data.std(dim=0)
            data = (data - mean[k]) / (std[k] + torch.finfo(data.dtype).eps)
            self.embeddings[k] = data.cpu()
        return dict(mean=mean, std=std)
    
    @torch.no_grad()
    @torch.inference_mode()
    def normalize(self, min_val={}, max_val={}, device=torch.device('cpu')):
        for k in tqdm(self.embeddings.keys(), desc='Normalizing Data'):
            data = self.embeddings[k].to(device)
            if k not in min_val.keys():
                min_val[k], _ = data.min(dim=0)
            if k not in max_val.keys():
                max_val[k], _ = data.max(dim=0)
            range_val = max_val[k] - min_val[k]
            range_val[range_val == 0] = 1.0 #avoid division by 0
            data = (data - min_val[k]) / range_val
            self.embeddings[k] = data.cpu()
        return dict(min_val=min_val, max_val=max_val)
    
    @property
    def storage(self):
        return sum(batch.element_size()*batch.numel() for batches in self.embeddings.values() for batch in batches)

@torch.no_grad()
@torch.inference_mode()
def compute_svdvals(loader:EmbeddingLoader, prefix:str, overwrite=True, device=torch.device('cpu'), dtype=None, escape_oom_cpu=False):
    pbar = tqdm(reversed(loader.embeddings), desc='Computing Sigmas')
    for name in pbar:
        # get data
        X = loader.embeddings[name]
        mem = X.element_size()*X.numel()
        pbar.set_postfix({'curr': name, 'mem':f'{mem/1e6:.1f}MB' if mem < 1e9 else f'{mem/1e9:.1f}GB'})
        
        # check if result exists 
        fname = f'{prefix}svdvals-{name}.pckl'
        if overwrite is False and os.path.isfile(fname):
            loader.embeddings[name] = None
            continue

        try: # compute on GPU and potentially resort to CPU 
            svdvals = torch.linalg.svdvals(X.to(device, dtype=dtype)).detach().cpu()
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise e
            tqdm.write(f'Could not compute svdvals for {name}: '+str(e))
            #warn(f'Could not compute svdvals for {name}: '+str(e))
            svdvals = torch.Tensor([float('nan')])
            if escape_oom_cpu:
                svdvals = torch.linalg.svdvals(X.to(dtype=dtype))

        with open(fname, 'wb') as f:
            pickle.dump(svdvals, f)

        loader.embeddings[name] = None
        gc.collect()


@torch.no_grad()
@torch.inference_mode()
def compute_svdcluster(loader:EmbeddingLoader, prefix:str, overwrite=True, device=torch.device('cpu'), dtype=None, escape_oom_cpu=False):

    pbar = tqdm(reversed(loader.embeddings), desc='Computing SVDs')
    for name in pbar:
        # get data
        y = loader.targets
        X = loader.embeddings[name]
        mem = X.element_size()*X.numel()
        pbar.set_postfix({'curr': name, 'mem':f'{mem/1e6:.1f}MB' if mem < 1e9 else f'{mem/1e9:.1f}GB'})
        
        # check if result exists 
        fname = f'{prefix}SVDg-{name}.pckl'
        if overwrite is False and os.path.isfile(fname):
            loader.embeddings[name] = None
            continue

        ### Compute globally, within-cluster and between-cluster centered data
        C, N, D = len(y.unique()), *X.shape
        Nc = torch.Tensor([torch.sum(y==c) for c in y.unique()]).to(device)

        Xw = torch.zeros(N, D).to(device)
        Xb = torch.zeros(C, D).to(device)
        for idx, c in enumerate(y.unique()):
            Xw[y==c] = (X[y==c] - X[y==c].mean(dim=0)) / sqrt(N-1)
            Xb[idx] = (X[y==c].mean(dim=0) - X.mean(dim=0)) * torch.sqrt(Nc[idx]) / sqrt(N-1)
        Xg = (X - X.mean(dim=0)) / sqrt(N-1)

        #Cov_g = Xg.T @ Xg 
        #Cov_w = Xw.T @ Xw 
        #Cov_b = Xb.T @ Xb 
        #assert torch.dist(Cov_g, Cov_w+Cov_b, p=float('inf')) < eps
        #Xr = torch.linalg.pinv(Cov_w) @ Cov_b
        #Sr = torch.linalg.svdvals(Xr)

        try: 
            SVDg = torch.linalg.svd(Xg.to(device, dtype=dtype), full_matrices=False)
            SVDw = torch.linalg.svd(Xw.to(device, dtype=dtype), full_matrices=False)
            SVDb = torch.linalg.svd(Xb.to(device, dtype=dtype), full_matrices=False)

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise e
            warn(f'Could not compute svdvals for {name}: '+str(e))
            SVDg = None
            SVDw = None
            SVDb = None
            
            if escape_oom_cpu:
                SVDg = torch.linalg.svd(Xg.to(dtype=dtype), full_matrices=False)
                SVDw = torch.linalg.svd(Xw.to(dtype=dtype), full_matrices=False)
                SVDb = torch.linalg.svd(Xb.to(dtype=dtype), full_matrices=False)

        #U_g, S_g, Vh_g = SVDg
        #U_w, S_w, Vh_w = SVDw
        #U_b, S_b, Vh_b = SVDb
        #Xr = (torch.diag(S_w**(-2)) @ Vh_w)  @  (Vh_b.T @ torch.diag(S_b**(2)))
        #Sr = torch.linalg.svdvals(Xr)

        with open(f'{prefix}SVDg-{name}.pt', 'wb') as f:
            torch.save(SVDg, f)
        with open(f'{prefix}SVDw-{name}.pt', 'wb') as f:
            torch.save(SVDw, f)
        with open(f'{prefix}SVDb-{name}.pt', 'wb') as f:
            torch.save(SVDb, f)

        loader.embeddings[name] = None
        gc.collect()


@torch.no_grad()
@torch.inference_mode()
def compute_probes(loader:EmbeddingLoader, valid_loader:EmbeddingLoader, prefix:str, overwrite=True, device=torch.device('cpu'), dtype=None, escape_oom_cpu=False):
    from dinopl.probing import Analysis, LinearAnalysis, KNNAnalysis, LogRegAnalysis, LinDiscrAnalysis
    pbar = tqdm(reversed(loader.embeddings), desc='Probing')
    for name in pbar:
        # get data
        y = loader.targets
        X = loader.embeddings[name]
        y_val = valid_loader.targets
        X_val = valid_loader.embeddings[name]
        mem = X.element_size()*X.numel()
        pbar.set_postfix({'curr': name, 'mem':f'{mem/1e6:.1f}MB' if mem < 1e9 else f'{mem/1e9:.1f}GB'})
        
        # check if result exists 
        fname = f'{prefix}probe-{name}.pckl'
        if overwrite is False and os.path.isfile(fname):
            loader.embeddings[name] = None
            valid_loader.embeddings[name] = None
            continue
        
        acc = {}
        C, N, D = len(y.unique()), *X.shape
        train_data = list(zip(X.chunk(ceil(N/256)), y.chunk(ceil(N/256))))
        valid_data = list(zip(X_val.chunk(ceil(N/256)), y_val.chunk(ceil(N/256))))
        for clf in [LinearAnalysis(20), KNNAnalysis(20), LogRegAnalysis(), LinDiscrAnalysis()]:
            clf:Analysis = clf
            
            try: # try clf on GPU
                clf.prepare(n_features=D, n_classes=C, device=device) #LogReg will ignore device
                clf.train(train_data)
                acc[type(clf).__name__] = clf.valid(valid_data)
            except RuntimeError as e:
                if 'out of memory' not in str(e):
                    raise e
                warn(f'Could not compute clf for {name}: '+str(e))
                if escape_oom_cpu:
                    clf.prepare(n_features=D, n_classes=C, device=torch.device('cpu')) #LogReg will ignore device
                    clf.train(train_data)
                    acc[type(clf).__name__] = clf.valid(valid_data)
        
        with open(f'{prefix}probes-{name}.pckl', 'wb') as f:
            pickle.dump(acc, f)

        loader.embeddings[name] = None
        valid_loader.embeddings[name] = None
        gc.collect()
    return acc

def evaluate_ckpt(fname, args):
    # prepare results
    dname = os.path.splitext(fname)[0]
    os.makedirs(dname, exist_ok=True)
    prefix = f'{dname}/'

    # load model 
    model = load_model(fname).to(args.device)
    dl, valid_dl = load_data(fname, batchsize=256, num_workers=4, pin_memory=(args.device.type=='cuda'))
    if args.ds_split == 'valid':
        prefix = f'{prefix}valid-'
        dl = valid_dl
    
    # load embeddings
    loader = EmbeddingLoader(model=model, rootname=f'{args.model}.enc', layers=args.track_layers)
    loader.load_data(dl, device=args.device)


    # preprocess
    if args.preprocess == 'center':
        prefix = f'{prefix}centd-'
        data_stats = loader.center()
    if args.preprocess == 'standardize':
        prefix = f'{prefix}stdd-'
        data_stats = loader.standardize()
    if args.preprocess == 'normalize':
        prefix = f'{prefix}normd-'
        data_stats = loader.normalize()

    if args.dtype is not None:
        prefix = f'{prefix}{str(args.dtype).split(".")[-1]}-'

    # load values
    if args.compute == 'svdvals':
        compute_svdvals(loader, prefix=prefix, overwrite=args.overwrite, device=args.device, 
                        dtype=args.dtype, escape_oom_cpu=args.escape_oom_cpu)
    if args.compute == 'svdcluster':
        compute_svdcluster(loader, prefix=prefix, overwrite=args.overwrite, device=args.device, 
                        dtype=args.dtype, escape_oom_cpu=args.escape_oom_cpu)
    
    if args.compute == 'probes':
        if args.ds_split == 'valid':
            raise ValueError('For fitting of probes, the training set needs to be used.')
        valid_loader = EmbeddingLoader(model=model, rootname=f'{args.model}.enc', layers=args.track_layers)
        valid_loader.load_data(valid_dl, device=args.device)
        if args.preprocess == 'center':
            valid_loader.center(**data_stats)
        if args.preprocess == 'standardize':
            valid_loader.standardize(**data_stats)
        if args.preprocess == 'normalize':
            valid_loader.normalize(**data_stats)

        compute_probes(loader, valid_loader, prefix=prefix, overwrite=args.overwrite, device=args.device, 
                        dtype=args.dtype, escape_oom_cpu=args.escape_oom_cpu)


def main(args):
    sweep = api.sweep(f'safelix/DINO/sweeps/{args.sweep}')
    if 'best' in args.runs:
        runs = [sweep.best_run()]
    elif 'all' in args.runs:
        runs = list(sweep.runs)
        runs = tqdm(runs, desc='Evaluating Runs')
    else:
        runs = [api.run(f'safelix/DINO/runs/{run}') for run in args.runs]
        runs = tqdm(runs, desc='Evaluating Runs')

    for run in runs:
        if args.ckpt == 'last':
            fname = 'last*.ckpt'
        if args.ckpt == 'probe_student':
            fname = 'epoch=*-probe_student=*.ckpt'
        if args.ckpt == 'loss_max':
            fname = 'epoch=*-loss_max=*.ckpt'
        if args.ckpt == 'rank_min':
            fname = 'epoch=*-rank_min=*.ckpt'

        fname = os.path.join(os.environ['DINO_RESULTS'], 'DINO', run.id, fname)
        fname = (glob(fname) + [None])[0]
        print(f'Evaluating {fname}')
        evaluate_ckpt(fname, args)
    

if __name__ == '__main__':
    def str2bool(s:str):
        if s.lower() not in ['on', 'true', '1'] + ['off', 'false', '0']:
            raise argparse.ArgumentTypeError('invalid value for a boolean flag')
        return s.lower() in ['on', 'true', '1']
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, default='4zyei965')
    parser.add_argument('--runs', type=str, nargs='*', default=['best'])
    parser.add_argument('--ckpt', type=str, default='probe_student',
                        choices=['last', 'probe_student', 'loss_max', 'rank_min'])
    parser.add_argument('--model', type=str, default='student',
                        choices=['student', 'teacher'])
    parser.add_argument('--track_layers', type=str, default='root',
                        choices=['root', 'leafs'])    
    parser.add_argument('--ds_split', default='train',
                        choices=['train', 'valid'])
    parser.add_argument('--preprocess', default=None,
                        choices=[None, 'center', 'standardize', 'normalize'])
    parser.add_argument('--compute', default='svdvals',
                        choices=['svdvals', 'svdcluster', 'probes'])
    parser.add_argument('--overwrite', type=str2bool, default=False)
    parser.add_argument('--dtype', default=None,
                        choices=['float16, float32', 'float64'])
    parser.add_argument('--device', type=str, default='-1')
    parser.add_argument('--escape_oom_cpu', type=str2bool, default=False)
    args = parser.parse_args()

    if args.dtype is not None:
        args.dtype = getattr(torch, args.dtype)

    if args.device == '-1':
        args.device = pick_single_gpu()
    args.device = torch.device(args.device)

    main(args)