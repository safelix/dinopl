import copy
from collections import OrderedDict
import math
from typing import List, Type

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from . import utils as U
from .scheduling import *

__all__ = [
    'DINOHead',
    'DINOModel',
    'MultiCropAugmentation',
    'DINOTeacherUpdater',
    'DINO',
]

class DINOHead(nn.Module):
    def __init__(self,
        embed_dim:int ,
        out_dim:int,
        hidden_dims:List[int] = [2048, 2048],
        l2_bottleneck_dim:int = 256,
        use_bn:bool = False, 
        act_fn:str = 'GELU',
        temp:float = 1.0,     
        cent:float = 0.0,
        cmom:float = torch.nan,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.temp = temp
        self.cmom = cmom
        self.register_buffer('cent', torch.full((out_dim,), cent))

        # multi-layer perceptron classification head
        layers, dims = OrderedDict(), [embed_dim] + hidden_dims
        for idx, (i, o) in enumerate(zip(dims[:-1], dims[1:])):
            layers[f'layer{idx}'] = U.mlp_layer(i, o, act_fn, use_bn)
        
        if l2_bottleneck_dim is None or l2_bottleneck_dim <= 0:
            layers['output'] = nn.Linear(dims[-1], out_dim)
        else:
            layers['bottleneck'] = U.L2Bottleneck(dims[-1], l2_bottleneck_dim, out_dim)

        self.mlp = nn.Sequential(layers)  # build mlp

        # Initallization with trunc_normal()
        self.mlp.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, U.WeightNormalizedLinear):
            return # explicitely do not apply to weight normalized
        if isinstance(m, nn.Linear):
            U.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x:torch.Tensor, update_cent:bool=False):
        # [n_crops, n_batches, embed_dim]
        # -> [n_crops, n_batches, out_dim]

        projections = self.mlp(x)
        batch_cent = torch.mean(projections, dim=[0,1]).detach()
        logits = (projections - self.cent) / self.temp

        # update centering after it is applied to projections
        if update_cent and not math.isnan(self.cmom): # only if activated
            self.cent = self.cent * self.cmom  + batch_cent * (1 - self.cmom)
        
        out = dict(logits=logits, projections=projections)
        return out
        

class DINOModel(nn.Module):
    def __init__(self,
        enc: nn.Module,
        head: DINOHead,
        ):
        super().__init__()
        self.enc = enc
        self.embed_dim = enc.embed_dim
        self.head = head
        self.out_dim = head.out_dim
        self.crops = {'name':[], 'idx':[]}

    def forward(self, crop_batches, **kwargs):
        # [n_crops, n_batches, n_channels, height, width]
        if not isinstance(crop_batches, list):
            crop_batches = [crop_batches]
        
        # encode batch for every crop and merge
        # -> [n_crops, n_batches, embed_dim]
        embeddings = list(map(self.enc, crop_batches))
        embeddings = torch.stack(embeddings, dim=0)     

        # compute outputs from embeddings
        # -> [n_crops, n_batches, out_dim]
        out = self.head(embeddings, **kwargs)
        out['embeddings'] = embeddings
        return out



mc_spec = [
        {'name':'global1', 'out_size':244, 'min_scale':0.4, 'max_scale':1, 'teacher':True, 'student':True},
        {'name':'global2', 'out_size':244, 'min_scale':0.4, 'max_scale':1, 'teacher':True, 'student':True},
    ]
class MultiCropAugmentation(nn.Module):
    f'''
    Takes a list of crop specifications.
    Example:
    {mc_spec}
    '''

    def __init__(self, spec:List[dict], per_crop_transform=torch.nn.Identity):
        super().__init__()
        self.spec = spec
        self.per_crop_transform = per_crop_transform

        self.transforms = {}
        for crop_spec in self.spec:
            name = crop_spec['name']
            transform = self.get_transform(crop_spec)   
            self.transforms[name] = transform
            self.__dict__[name] = transform

    def get_transform(self, crop_spec):
        return transforms.RandomResizedCrop(
                    size = crop_spec['out_size'],
                    scale = (crop_spec['min_scale'], crop_spec['max_scale'])
                )

    def forward(self, img):
        # [height, width, (n_channels)]
        # -> [n_crops, n_channels, height, width]
        crops = []
        for name, transform in self.transforms.items():
            crop = transform(img)
            crop = self.per_crop_transform(crop)
            crops.append(crop)
        return crops


class DINOTeacherUpdater(pl.Callback):
    def __init__(self, mode: str = 'ema', mom: float = 0.996 ):
        if mode == 'no_update':
            pass
        elif mode == 'ema':
            self.mom = mom
            self.on_train_batch_end = self.ema
        elif mode == 'prev_epoch':
            self.on_train_epoch_end = self.copy 
        else:
            raise RuntimeError('Unkown teacher update mode.')

    def ema(self, _:pl.Trainer, dino: pl.LightningModule, *args):
        # TODO: should the BN buffers (encoder only? or everything except centering?) also be updated?
        for p_s, p_t in zip(dino.student.parameters(), dino.teacher.parameters()):
            p_t.data = self.mom * p_t.data + (1 - self.mom) * p_s.data

    def copy(self, _:pl.Trainer, dino: pl.LightningModule, *args):
        for p_s, p_t in zip(dino.student.parameters(), dino.teacher.parameters()):
            p_t.data = p_s.data
        

class DINO(pl.LightningModule):
    def __init__(self,
        mc:MultiCropAugmentation,
        model:DINOModel,
        t_mode:str = 'ema',
        t_eval:bool = False,
        s_mode:str = 'self-supervised',
        t_mom:Schedule = CosSched(0.996, 1),
        t_cmom:Schedule = ConstSched(0.9),
        s_cmom:Schedule = ConstSched(torch.nan),
        t_temp:Schedule = LinWarmup(0.04, 0.04, 0),
        s_temp:Schedule = ConstSched(0.1),
        loss:str = 'CE',
        loss_pairing:str = 'opposite',
        opt:Type[optim.Optimizer] = optim.AdamW,
        opt_lr:Schedule = None,
        opt_wd:Schedule = None,
        wn_freeze_epochs = 1,
    ):
        super().__init__()
        self.t_eval = t_eval
        self.embed_dim = model.embed_dim
        self.out_dim = model.out_dim
        self.wn_freeze_epochs = wn_freeze_epochs

        # check if all string inputs are valid
        if t_mode not in ['ema', 'prev_epoch', 'no_update']:
            raise RuntimeError(f'Teacher update mode \'{t_mode}\' not supported.')
        self.t_mode = t_mode

        if s_mode not in ['supervised', 'self-supervised']:
            raise RuntimeError(f'Student update mode \'{s_mode}\' not supported.')
        self.s_mode = s_mode

        if loss not in ['CE', 'KL', 'H_pred']:
            raise RuntimeError(f'Loss \'{loss}\' not supported.')
        self.loss = loss

        if loss_pairing not in ['all', 'same', 'opposite']:
            raise RuntimeError(f'Pairing strategy \'{loss_pairing}\' not supported.')
        self.loss_pairing = loss_pairing
        
        # initiallize student and teacher with same params
        self.student = model
        self.teacher = copy.deepcopy(model)
      
        # prepare teacher in evaluation mode
        #self.teacher.eval() # TODO: will this stay in eval mode? should this be in eval mode?
        for p in self.teacher.parameters():
            p.requires_grad = False

        # store crops
        for idx, crop in enumerate(mc.spec):
            if crop['teacher']:
                self.teacher.crops['name'].append(crop['name'])
                self.teacher.crops['idx'].append(idx)
            if crop['student']:
                self.student.crops['name'].append(crop['name'])
                self.student.crops['idx'].append(idx)

        # prepare schedulers for optimization procedure
        self.scheduler = Scheduler()

        # configure teacher updater
        self.t_updater = DINOTeacherUpdater(self.t_mode)
        self.scheduler.add(self.t_updater, 'mom', t_mom)
        
        # configure teacher temperature and centering
        self.scheduler.add(self.teacher.head, 'cmom', t_cmom)
        self.scheduler.add(self.student.head, 'cmom', s_cmom)
        self.scheduler.add(self.teacher.head, 'temp', t_temp)
        self.scheduler.add(self.student.head, 'temp', s_temp)

        # configure optimizer, learning rate & weight decay       
        params = list(self.student.named_parameters()) # generator -> list
        self.optimizer = opt([
            {'params':[p for n,p in params if not U.is_bias(n,p)]},
            {'params':[p for n,p in params if U.is_bias(n,p)]}])
        
        if opt_lr is not None:
            self.scheduler.add(self.optimizer.param_groups[0], 'lr', opt_lr)
            self.scheduler.add(self.optimizer.param_groups[1], 'lr', opt_lr)
        if opt_wd is not None: # only regularize weights but not biases
            self.scheduler.add(self.optimizer.param_groups[0], 'weight_decay', opt_wd)

        print(f'Init optimizer: {len(self.optimizer.param_groups)} paramgroups of sizes', 
            [len(group['params']) for group in self.optimizer.param_groups])
    
    
    def configure_optimizers(self):
        return self.optimizer

    def configure_callbacks(self):
        return [self.scheduler, self.t_updater]

    def on_fit_start(self) -> None:
        # move scheduler and updater to the front
        for idx, cb in enumerate(self.trainer.callbacks):
            if isinstance(cb, Scheduler):
                self.trainer.callbacks.insert(0, self.trainer.callbacks.pop(idx))
            if isinstance(cb, DINOTeacherUpdater):
                self.trainer.callbacks.insert(1, self.trainer.callbacks.pop(idx))

        print('Order of Callbacks: ')
        for idx, cb in enumerate(self.trainer.callbacks, 1):
            print(f' {idx}. {type(cb).__name__}', flush=True)

        # linear scaling rule: schedule is by refernce... only scale once
        bs = self.trainer.train_dataloader.loaders.batch_size
        self.scheduler.get(self.optimizer.param_groups[0], 'lr').ys *=  bs / 256
    

    def multicrop_loss(self, pred_logits: torch.Tensor, targ_logits: torch.Tensor = None, targ_labels: torch.Tensor = None):
        # [n_crops, n_batches, out_dim]

        # compute (log) softmax and per-crop entropy for predictions
        log_preds = F.log_softmax(pred_logits, dim=-1)
        preds = torch.exp(log_preds)
        H_preds = U.entropy(preds, log_preds) # [n_crops, n_batches]
        
        if targ_logits is not None and targ_labels is None:
            # compute (log) softmax and per-crop entropy for targets
            log_targs = F.log_softmax(targ_logits, dim=-1)
            targs = torch.exp(log_targs)
            H_targs = U.entropy(targs, log_targs) # [n_crops, n_batches]
        
        elif targ_labels is not None and targ_logits is None:
            # compute softmax and per-crop entropy for targets
            targs = F.one_hot(targ_labels).unsqueeze(0).expand(len(self.teacher.crops), -1, -1)
            H_targs = torch.zeros(*targs.shape[:2], device=self.device) # entropy of one-hot is zero
        else:
            raise RuntimeError('Please specify either targ_logits or targ_one_hot.')

        # compute pairwise losses
        KLs, CEs = [], []
        for i_stud, targ, H_targ in zip(self.teacher.crops['idx'], targs, H_targs):  
            for i_teach, log_pred in zip(self.student.crops['idx'], log_preds):
                
                # case 'oposite': don't pair same views
                if self.loss_pairing == 'opposite' and i_stud == i_teach: 
                    continue
                
                # case 'matching': don't pair oposite views
                if self.loss_pairing == 'same' and i_stud != i_teach:
                    continue

                # case 'all': pair all views

                CEs.append(U.cross_entropy(log_pred, targ))         # list of [n_batches]
                KLs.append(CEs[-1] - H_targ) # U.kl_divergence()    # list of [n_batches]

        if len(CEs) == 0 or len(KLs) == 0:
            raise RuntimeError('No pairwise losses where computed.')

        CEs = torch.stack(CEs, dim=0) # [n_pairs, n_batches] 
        KLs = torch.stack(KLs, dim=0) # [n_pairs, n_batches]

        # aggregate losses
        out = {}
        out['CE'] = CEs.mean(dim=-1).mean(dim=-1) # compute mean of batches, than of all matched crops
        out['KL'] = KLs.mean(dim=-1).mean(dim=-1) # compute mean of batches, than of all matched crops
        out['H_pred'] = H_preds.mean(dim=-1).mean(dim=-1) # compute mean of batches, than of all matched crops
        out['H_preds'] = H_preds
        out['H_targs'] = H_targs
        return out

    def training_step(self, batch, batch_idx):
        batch, batch_labels = batch

        # generate teacher's targets and student's predictions
        # [n_crops, n_batches, n_channels, height, width]
        # -> [n_crops, n_batches, out_dim]

        if self.t_eval: # set teacher in evaluation mode
            self.teacher.eval() 
        with torch.no_grad(): # don't compute gradients of teacher predictions
            teacher_out = self.teacher([batch[i] for i in self.teacher.crops['idx']], update_cent=True)
        student_out = self.student([batch[i] for i in self.student.crops['idx']], update_cent=True)
        
        # compute multicrop loss
        if self.s_mode == 'self-supervised':
            out = self.multicrop_loss(student_out['logits'], targ_logits=teacher_out['logits'])
        else:
            out = self.multicrop_loss(student_out['logits'], targ_labels=batch_labels)

        # minimize CE loss
        out['loss'] = out[self.loss]        
        out['teacher'] = teacher_out
        out['student'] = student_out
        return out
            
    def validation_step(self, batch, batch_idx):
        batch, batch_labels = batch

        # generate teacher's targets and student's predictions
        # [n_crops, n_batches, n_channels, height, width]
        # -> [n_crops, n_batches, out_dim]
        teacher_out = self.teacher([batch[i] for i in self.teacher.crops['idx']])
        student_out = self.student([batch[i] for i in self.student.crops['idx']])
        
        # compute multicrop loss
        if self.s_mode == 'self-supervised':
            out = self.multicrop_loss(student_out['logits'], targ_logits=teacher_out['logits'])
        else:
            out = self.multicrop_loss(student_out['logits'], targ_labels=batch_labels)

        # minimize CE loss
        out['loss'] = out[self.loss]
        out['teacher'] = teacher_out
        out['student'] = student_out
        return out


    # Set gradients to `None` instead of zero to improve performance.
    def optimizer_zero_grad(self, epoch, batch_idx, opt:torch.optim.Optimizer, optimizer_idx):
        opt.zero_grad(set_to_none=True)

    def on_before_optimizer_step(self, *args):
        if hasattr(self.student.head.mlp, 'bottleneck'):
            wn = self.student.head.mlp.bottleneck.weightnorm
            if self.current_epoch < self.wn_freeze_epochs:
                for p in wn.parameters():
                    p.grad = None
