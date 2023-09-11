import math
from typing import Any, Dict, List, Optional, Type, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F

from models import Encoder

from . import utils as U
from .modules import L2Bottleneck, MLP, Linear
from .scheduling import *
from torchvision import transforms

__all__ = [
    'MultiCrop',
    'DINOHead',
    'DINOModel',
    'DINOTeacherUpdater',
    'DINO',
]

mc_spec = [
        {'name':'global1', 'out_size':244, 'min_scale':0.4, 'max_scale':1, 'teacher':True, 'student':True},
        {'name':'global2', 'out_size':244, 'min_scale':0.4, 'max_scale':1, 'teacher':True, 'student':True},
    ]
class MultiCrop(nn.Module):
    f'''
    Takes a list of crop specifications.
    Example:
    {mc_spec}
    '''

    def __init__(self, spec:List[dict], per_crop_transform=nn.Identity):
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


class DINOHead(nn.Module):
    def __init__(self,
        embed_dim:int ,
        out_dim:int,
        hidden_dims:List[int] = [2048, 2048],
        l2bot_dim:int = 256,
        l2bot_cfg:str = '-/lb/fn/wn/l/-',
        use_bn:bool = False, 
        act_fn:str = 'GELU',
        temp:float = 1.0,     
        cent:float = 0.0,
        cmom:float = torch.nan,
        init_method = 'trunc_normal',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.temp = temp
        self.cmom = cmom
        self.register_buffer('cent', torch.full((out_dim,), cent))
        self.init_method = init_method

        # multi-layer perceptron classification head
        dims = [embed_dim] + hidden_dims
        self.mlp = MLP(dims, act_fn=act_fn, use_bn=use_bn)  # build mlp

        if l2bot_dim is None or l2bot_dim <= 0:
            self.last_layer = Linear(dims[-1], out_dim)
        else:
            self.last_layer = L2Bottleneck(dims[-1], l2bot_dim, out_dim, l2bot_cfg)

    def reset_parameters(self, generator:torch.Generator=None):
        self.mlp.reset_parameters(method=self.init_method, generator=generator)
        self.last_layer.reset_parameters(method=self.init_method, generator=generator)

    def forward(self, x:torch.Tensor, update_cent:bool=False) -> Dict[str, torch.Tensor]:
        # [n_crops, n_batches, embed_dim]
        # -> [n_crops, n_batches, out_dim]

        projections = self.last_layer(self.mlp(x))
        batch_cent = torch.mean(projections, dim=[0,1]).detach()
        logits = (projections - self.cent) / self.temp

        # update centering after it is applied to projections
        if update_cent and not math.isnan(self.cmom): # only if activated
            self.cent = self.cent * self.cmom  + batch_cent * (1 - self.cmom)
        
        out = dict(logits=logits, projections=projections)
        return out
        

class DINOModel(nn.Module):
    def __init__(self,
        enc: Union[Encoder, nn.Module],
        head: DINOHead,
        ):
        super().__init__()
        self.enc = enc
        self.embed_dim = enc.embed_dim
        self.head = head
        self.out_dim = head.out_dim
        self.crops : Dict[str, List] = None
        self.return_dict = True

    def reset_parameters(self, generator:torch.Generator=None):
        self.enc.reset_parameters(generator=generator)
        self.head.reset_parameters(generator=generator)

    def forward(self, crop_batches, **kwargs) -> Dict[str, torch.Tensor]:
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

        if not self.return_dict:
            return out['logits']

        out['embeddings'] = embeddings
        return out


class DINOTeacherUpdater(pl.Callback):
    def __init__(self, mode: str = 'ema', mom: float = 0.996, update_every=1, update_bn=True):
        if mode == 'no_update':
            pass
        elif mode == 'ema':
            self.mom = mom
            self.on_train_batch_end = self.ema
        elif mode == 'prev_epoch':
            self.update_every = update_every
            self.on_train_epoch_end = self.copy 
        else:
            raise RuntimeError('Unkown teacher update mode.')
        self.update_bn = update_bn

    def ema(self, _:pl.Trainer, dino: 'DINO', *args):
        for p_s, p_t in zip(dino.student.parameters(), dino.teacher.parameters()):
            p_t.data = self.mom * p_t.data + (1 - self.mom) * p_s.data

        if not self.update_bn:
            return # update BN stat buffers if required
        for (n_s, m_s), (n_t, m_t) in zip(dino.student.named_modules(), dino.teacher.named_modules()):
            if isinstance(m_s, torch.nn.modules.batchnorm._NormBase) and n_s == n_t:
                m_t.running_mean.data = self.mom * m_t.running_mean.data + (1 - self.mom) * m_s.running_mean.data
                m_t.running_var.data = self.mom * m_t.running_var.data + (1 - self.mom) * m_s.running_var.data

    def copy(self, _:pl.Trainer, dino: 'DINO', *args):
        if not (dino.current_epoch % self.update_every == self.update_every - 1):
            return # skip if not update_every

        for p_s, p_t in zip(dino.student.parameters(), dino.teacher.parameters()):
            p_t.data = p_s.data
        
        if not self.update_bn: 
            return # update BN stat buffers if required
        for (n_s, m_s), (n_t, m_t) in zip(dino.student.named_modules(), dino.teacher.named_modules()):
            if isinstance(m_s, torch.nn.modules.batchnorm._NormBase) and n_s == n_t:
                m_t.running_mean.data = m_s.running_mean.data
                m_t.running_var.data = m_s.running_var.data

class DINO(pl.LightningModule):
    def __init__(self,
        mc_spec:List[Dict[str, Any]],
        student:DINOModel,
        teacher:DINOModel,
        s_mode:str = 'distillation',
        t_mode:str = 'ema',
        t_mom:Schedule = CosSched(0.996, 1),
        t_update_every:int = 1,
        t_bn_mode:str = 'from_data',
        t_eval:bool = False,
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
        assert(student is not teacher) # need to be different objects
        self.student = student
        self.teacher = teacher
        self.embed_dim = student.embed_dim
        self.out_dim = student.out_dim
        self.s_mode = s_mode
        self.t_mode = t_mode
        self.t_eval = t_eval
        self.loss = loss
        self.loss_pairing = loss_pairing
        self.wn_freeze_epochs = wn_freeze_epochs

        # check if all string inputs are valid
        if t_mode not in ['ema', 'prev_epoch', 'no_update']:
            raise RuntimeError(f'Teacher update mode \'{t_mode}\' not supported.')
        
        if t_bn_mode not in ['from_data', 'from_student']:
            raise RuntimeError(f'Teacher batchnorm update mode \'{t_bn_mode}\' not supported.')
        
        if (t_bn_mode=='from_student' and t_eval==False): # or (t_bn_mode=='from_data' and t_eval==True) :
            raise RuntimeError(f'Invalid configuration: t_bn_mode==\'{t_bn_mode}\' and t_eval=={t_eval}')

        if s_mode not in ['supervised', 'distillation']:
            raise RuntimeError(f'Student update mode \'{s_mode}\' not supported.')

        if loss not in ['CE', 'KL', 'H_pred', 'MSE']:
            raise RuntimeError(f'Loss \'{loss}\' not supported.')

        if loss_pairing not in ['all', 'same', 'opposite']:
            raise RuntimeError(f'Pairing strategy \'{loss_pairing}\' not supported.')

        # chose loss function
        self.multicrop_loss = self.multicrop_loss_clf
        if loss == 'MSE':
            self.multicrop_loss = self.multicrop_loss_reg

        # store 
        self.teacher.crops = {'name':[], 'idx':[]}
        self.student.crops = {'name':[], 'idx':[]}
        for idx, crop in enumerate(mc_spec):
            if crop['teacher']:
                self.teacher.crops['name'].append(crop['name'])
                self.teacher.crops['idx'].append(idx)
            if crop['student']:
                self.student.crops['name'].append(crop['name'])
                self.student.crops['idx'].append(idx)

        # prepare schedulers for optimization procedure
        self.scheduler = Scheduler()

        # configure teacher updater
        self.teacher.requires_grad_(False)
        self.t_updater = DINOTeacherUpdater(self.t_mode, 
                                update_every=t_update_every, 
                                update_bn=(t_bn_mode=='from_student'))
        self.scheduler.add(self.t_updater, 'mom', t_mom)
        
        # configure teacher temperature and centering
        self.scheduler.add(self.teacher.head, 'cmom', t_cmom)
        self.scheduler.add(self.student.head, 'cmom', s_cmom)
        self.scheduler.add(self.teacher.head, 'temp', t_temp)
        self.scheduler.add(self.student.head, 'temp', s_temp)

        # configure optimizer, learning rate & weight decay
        self.student.requires_grad_(True)       
        params = list(self.student.named_parameters()) # generator -> list
        self.optimizer = opt([
            {'params':[p for n,p in params if not U.is_bias(n,p)]},
            {'params':[p for n,p in params if U.is_bias(n,p)]}],
            lr=float('inf')) # learning rate will be overwritten by scheduler TODO: we could use torch.optim.optimizer.required
        
        if opt_lr is not None:
            self.scheduler.add(self.optimizer.param_groups[0], 'lr', opt_lr)
            self.scheduler.add(self.optimizer.param_groups[1], 'lr', opt_lr)
        if opt_wd is not None: # only regularize weights but not biases
            self.scheduler.add(self.optimizer.param_groups[0], 'weight_decay', opt_wd)
    
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
        #bs = self.trainer.train_dataloader.loaders.batch_size
        #self.scheduler.get(self.optimizer.param_groups[0], 'lr').ys *=  bs / 256
    
    def multicrop_loss_reg(self, pred_logits: torch.Tensor, targ_logits: torch.Tensor = None, targ_labels: torch.Tensor = None):
        # [n_crops, n_batches, out_dim]

        # take pred_logits as predictions
        preds = pred_logits
        
        if targ_logits is not None and targ_labels is None:
            # take targ_logits as target targets
            targs = targ_logits
 
        elif targ_labels is not None and targ_logits is None:
            targs = F.one_hot(targ_labels, self.out_dim).unsqueeze(0).expand(len(self.teacher.crops), -1, -1)
            #raise NotImplementedError('Regression losses are currently not supported for target labels, only logits.')
        else:
            raise RuntimeError('Please specify either targ_logits or targ_labels.')

        # compute pairwise losses
        MSEs = []
        for i_stud, targ, in zip(self.teacher.crops['idx'], targs):  
            for i_teach, pred in zip(self.student.crops['idx'], preds):
                
                # case 'oposite': don't pair same views
                if self.loss_pairing == 'opposite' and i_stud == i_teach: 
                    continue
                
                # case 'matching': don't pair oposite views
                if self.loss_pairing == 'same' and i_stud != i_teach:
                    continue

                # case 'all': pair all views

                MSEs.append(U.mean_squared_error(pred, targ))         # list of [n_batches]

        if len(MSEs) == 0:
            raise RuntimeError('No pairwise losses where computed.')

        MSEs = torch.stack(MSEs, dim=0) # [n_pairs, n_batches] 

        # aggregate losses
        out = {}
        out['MSE'] = MSEs.mean(dim=-1).mean(dim=-1) # compute mean of batches, than of all matched crops
        return out

    def multicrop_loss_clf(self, pred_logits: torch.Tensor, targ_logits: torch.Tensor = None, targ_labels: torch.Tensor = None):
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
            targs = F.one_hot(targ_labels, self.out_dim).unsqueeze(0).expand(len(self.teacher.crops), -1, -1)
            H_targs = torch.zeros(*targs.shape[:2], device=self.device) # entropy of one-hot is zero
        else:
            raise RuntimeError('Please specify either targ_logits or targ_labels.')

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

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        batch, batch_targets = batch

        # generate teacher's targets and student's predictions
        # [n_crops, n_batches, n_channels, height, width]
        # -> [n_crops, n_batches, out_dim]

        self.train()
        if self.t_eval: # set teacher in evaluation mode
            self.teacher.eval() 
        assert(self.student.training and self.teacher.training != self.t_eval)
            
        # gradient is not computed because of teacher.requires_grad_(False)
        teacher_out = self.teacher([batch[i] for i in self.teacher.crops['idx']], update_cent=True)
        student_out = self.student([batch[i] for i in self.student.crops['idx']], update_cent=True)
        assert(teacher_out['logits'].grad_fn is None)
        assert(student_out['logits'].grad_fn is not None)

        # compute multicrop loss
        if self.s_mode == 'distillation':
            out = self.multicrop_loss(student_out['logits'], targ_logits=teacher_out['logits'])
        elif batch_targets.dim() == 1:
            out = self.multicrop_loss(student_out['logits'], targ_labels=batch_targets)
        elif batch_targets.dim() == 2:
            out = self.multicrop_loss(student_out['logits'], targ_logits=batch_targets)

        # minimize CE loss
        out['loss'] = out[self.loss]        
        out['teacher'] = teacher_out
        out['student'] = student_out
        return out
    
            
    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        batch, batch_targets = batch

        self.eval()
        assert(not self.student.training and not self.teacher.training)

        # generate teacher's targets and student's predictions
        # [n_crops, n_batches, n_channels, height, width]
        # -> [n_crops, n_batches, out_dim]
        teacher_out = self.teacher([batch[i] for i in self.teacher.crops['idx']])
        student_out = self.student([batch[i] for i in self.student.crops['idx']])
        
        # compute multicrop loss
        if self.s_mode == 'distillation':
            out = self.multicrop_loss(student_out['logits'], targ_logits=teacher_out['logits'])
        elif batch_targets.dim() == 1:
            out = self.multicrop_loss(student_out['logits'], targ_labels=batch_targets)
        elif batch_targets.dim() == 2:
            out = self.multicrop_loss(student_out['logits'], targ_logits=batch_targets)

        # minimize CE loss
        out['loss'] = out[self.loss]
        out['teacher'] = teacher_out
        out['student'] = student_out
        return out


    # Set gradients to `None` instead of zero to improve performance.
    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def on_before_optimizer_step(self, *args):
        if self.current_epoch < self.wn_freeze_epochs:
            if getattr(self.student.head.last_layer, 'lin2', None) is not None:
                self.student.head.last_layer.lin2.zero_grad(True)