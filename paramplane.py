
import argparse
import copy
import os
from math import sqrt
from typing import Dict

import torch
import submitit
from tqdm import tqdm

from configuration import (Configuration, create_mc_spec, get_encoder,
                           init_student_teacher)
from dinopl import DINO, DINOHead, DINOModel
import dinopl.utils as U

from dinopl.probing import Prober, LinearAnalysis, KNNAnalysis


def load_model(identifier:str) -> DINOModel:
    ckpt_path, name = identifier.split(':')
    run_path = os.path.dirname(ckpt_path)

    # get configuration and prepare model
    config = Configuration.from_json(os.path.join(run_path, 'config.json'))
    config.mc_spec = create_mc_spec(config)
    enc = get_encoder(config)()
    config.embed_dim = enc.embed_dim
    head = DINOHead(config.embed_dim, config.out_dim, 
        hidden_dims=config.hid_dims, 
        l2bot_dim=config.l2bot_dim, 
        l2bot_cfg=config.l2bot_cfg,
        use_bn=config.mlp_bn,
        act_fn=config.mlp_act)
    student = DINOModel(enc, head)
    teacher = copy.deepcopy(student)

    # load DINO checkpoint
    dino = DINO.load_from_checkpoint(ckpt_path, map_location='cpu', mc_spec=config.mc_spec, student=student, teacher=teacher)

    # init if required by .init() suffix
    if name.endswith('.init()'):
        student, teacher = init_student_teacher(config, student)
        dino.student = student
        dino.teacher = teacher

    if name.startswith('teacher'):
        return dino.teacher
    
    if name.startswith('student'):
        return dino.student

    raise ValueError(f'Unkown name \'{name}\', should be either \'teacher\' or \'student\'.')


class ParamProjector():
    def __init__(self, vec0:torch.Tensor, vec1:torch.Tensor, vec2:torch.Tensor, center=None, scale=None) -> None:
        if center not in {None, '', 'mean', 'minnorm'}:
            raise ValueError(f'Unkown option \'{center}\' for argument \'center\'.')
        if scale not in {None, '', 'l2_ortho', 'rms_ortho'}:
            raise ValueError(f'Unkown option \'{scale}\' for argument \'scale\'.')
        self.center = center
        self.scale = scale
        self.dim = vec0.numel()

        self.affine:torch.Tensor = vec0
        if self.center == 'mean':
            self.affine = (vec0 + vec1 + vec2) / 3

        self.basis = torch.stack([vec1, vec2], dim=1)
        self.basis = self.basis - self.affine.unsqueeze(1)

        if self.center == 'minnorm':
            offset = self.basis @ torch.linalg.lstsq(self.basis, -self.affine).solution # origin projected to plane relative to affine
            self.affine = self.affine + offset
            self.basis = self.basis - offset.unsqueeze(1)

        self.affine_inv = torch.linalg.lstsq(self.basis, self.affine).solution
        if self.scale in {'l2_ortho', 'rms_ortho'}:
            self.basis:torch.Tensor = torch.linalg.svd(self.basis, full_matrices=False).U
            self.affine_inv = self.basis.T @ self.affine

        #self.affine_inv = torch.zeros_like(self.affine_inv)

    def project(self, vec:torch.Tensor, is_position=True) -> torch.Tensor:
        if is_position:
            vec = vec - self.affine

        if self.scale in {'l2_ortho', 'rms_ortho'}:
            coord = self.basis.T @ vec
        else:
            coord = torch.linalg.lstsq(self.basis, vec).solution

        if self.scale == 'rms_ortho': # rescale to preserve rms instead of norm
            coord = coord / sqrt(self.dim) * sqrt(2)
        return coord

    def map(self, coord:torch.Tensor, is_position=True) -> torch.Tensor:
        if self.scale == 'rms_ortho': # rescale to preserve rms instead of norm
            coord = coord / sqrt(2) * sqrt(self.dim)

        vec = self.basis @ coord
        
        if is_position:
            vec = vec + self.affine
        return vec

    def error(self, vec:torch.Tensor, p=2):
        diff = vec - self.map(self.project(vec))
        p = float(p) if p=='inf' else p
        return diff.square().mean().sqrt() if p=='rms' else diff.norm(p=p)
        
    def __call__(self, inp:torch.Tensor, is_direction=False) -> torch.Tensor:
        if inp.shape[0] == self.dim:
            return self.project(inp, is_direction)
        if inp.shape[0] == 2:
            return self.map(inp, is_direction)
        raise ValueError('Cannot infer whether to project or map input.')


def get_limits(data:torch.Tensor, aspectlim=(3/4, 1/1), margin=0.05, round_to_multiple=None):
    xmin, ymin = data.min(dim=0).values
    xmax, ymax = data.max(dim=0).values

    # enforce aspect ratio limits
    xrange = xmax - xmin
    yrange = ymax - ymin
    xcenter = xmin + xrange / 2
    ycenter = ymin + yrange / 2
    
    if yrange / xrange < aspectlim[0]:
        newrange = aspectlim[0] * xrange
        ymin = ycenter - newrange * (ycenter - ymin)/yrange
        ymax = ycenter + newrange * (ymax - ycenter)/yrange
        yrange = newrange

    if yrange / xrange > aspectlim[1]:
        newrange = aspectlim[1] * yrange
        xmin = ycenter - newrange * (ycenter - xmin)/xrange
        xmax = ycenter + newrange * (xmax - ycenter)/xrange
        yrange = newrange

    # add margins
    xmin = xcenter - (1+2*margin) * (xcenter - xmin)
    ymin = ycenter - (1+2*margin) * (ycenter - ymin)
    xmax = xcenter + (1+2*margin) * (xmax - xcenter)
    ymax = ycenter + (1+2*margin) * (ymax - ycenter)

    # round
    if round_to_multiple is None:
        round_to_multiple = 10**(data.std().log10().floor() - 1)
    if round_to_multiple > 0:
        xmin = (xmin / round_to_multiple).floor() * round_to_multiple
        xmax = (xmax / round_to_multiple).ceil() * round_to_multiple
        ymin = (ymin / round_to_multiple).floor() * round_to_multiple
        ymax = (ymax / round_to_multiple).ceil() * round_to_multiple

    return (float(xmin), float(xmax)), (float(ymin), float(ymax))


def eval_coord(coord:torch.Tensor, args):

    device = torch.device('cpu') if args['force_cpu'] else U.pick_single_gpu()    

    # make ParamProjector from and plane descriptor
    model0 = load_model(args['vec0']).to(device=device)
    model1 = load_model(args['vec1']).to(device=device)
    model2 = load_model(args['vec2']).to(device=device)
    
    P = ParamProjector(
        vec0=U.module_to_vector(model0),
        vec1=U.module_to_vector(model1),
        vec2=U.module_to_vector(model2),
        center=args['projector_center'],
        scale=args['projector_scale']
    )

    # get vector and model from coordinate
    vec = P(coord.to(device))
    model = copy.deepcopy(model0)
    U.vector_to_module(vec, model)

    # make experiments and store results
    out = {}
    out['coord'] = coord
    out['l2norm'] = vec.norm(p=2)

    return {k: v.cpu() for k,v in out.items()}


def main(args):
    dir = os.path.join(os.environ['DINO_RESULTS'], 'paramplane')

    X = torch.arange(args['xmin'], args['xmax'] + args['stepsize'], args['stepsize'])
    Y = torch.arange(args['ymin'], args['ymax'] + args['stepsize'], args['stepsize'])
    coords = torch.cartesian_prod(X, Y)
    print(f'Do you want to evaluate {coords.shape[0]} coordinates?')
    response = input()
    if response.lower() not in {'y', 'yes'}:
        return

    # start executor
    executor = submitit.AutoExecutor(folder=dir)
    executor.update_parameters(
        slurm_cpus_per_task=4,
        slurm_time=4,
        slurm_mem_per_cpu=4096,
        slurm_gpus_per_node=1)

    jobs = executor.map_array(eval_coord, coords, len(coords) * [args])

    out:Dict[str, torch.Tensor] = {}
    for idx, job in enumerate(tqdm(jobs)):
        res:Dict[str, torch.Tensor] = job.results()[0]

        for key, val in res.items():
            if key not in out.keys():
                out[key] = torch.zeros((coords.shape[0], *val.shape))
            out[key][idx] = val

    for key, val in out.items():
        fname = os.path.join(dir, f'{key}.pt')
        print(f'Saving {fname}')
        torch.save(val, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('vec0', type=str) # results/DINO/22abl14o/last.ckpt:teacher   # alpha=0
    parser.add_argument('vec1', type=str) # results/DINO/22abl14o/last.ckpt:student   # alpha=0
    parser.add_argument('vec2', type=str) # results/DINO/3mtlpc13/last.ckpt:student   # alpha=1

    parser.add_argument('--projector_center', choices={'', 'mean', 'minnorm'}, default='minnorm')
    parser.add_argument('--projector_scale', choices={'', 'l2_ortho', 'rms_ortho'}, default='l2_ortho')

    parser.add_argument('--xmin', type=float, default=None)
    parser.add_argument('--xmax', type=float, default=None)
    parser.add_argument('--ymin', type=float, default=None)
    parser.add_argument('--ymax', type=float, default=None)

    parser.add_argument('--stepsize', type=float)
    args = vars(parser.parse_args())
    main(args)

    
       

    
