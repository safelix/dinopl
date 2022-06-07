from typing import Dict, List, Type, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torchvision import transforms

import my_utils as U
from scheduling import *

__all__ = [
    'DINOHead',
    'DINOModel',
    'MultiCropAugmentation',
    'DINOTeacherUpdate',
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
        temp:float = 1,     
        cent:float = 0,
        cmom:float = None,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.temp = temp
        self.cent = torch.full((out_dim,), cent)
        self.cmom = None if cmom is None else nn.Parameter(cmom, requires_grad=False)

        # multi-layer perceptron classification head
        layers, dims = [], [embed_dim] + hidden_dims
        for i, o in zip(dims[:-1], dims[1:]):
            layers.append(U.mlp_layer(i, o, act_fn, use_bn))
        
        if l2_bottleneck_dim is None: # last layer
            layers.append(nn.Linear(dims[-1], out_dim))
        else:
            layers.append(
                U.l2_bottleneck(dims[-1], l2_bottleneck_dim, out_dim))

        self.mlp = nn.Sequential(*layers)  # build mlp
        self.log_softmax = nn.LogSoftmax()

        # Initallization with trunc_normal()
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                U.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x:torch.Tensor):
        # [n_crops, n_batches, embed_dim]
        # -> [n_crops, n_batches, out_dim]

        logits = self.mlp(x)
        if self.cmom:
            self.cent = self.cmom * self.cent + (1-self.cmom) * torch.mean(logits)
        y = self.log_softmax(logits - self.cent / self.temp)
        return y
        

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
        self.crops:Dict[str, Union[List[str],List[int]]] = None

    def forward(self, crop_batches):
        # [n_crops, n_batches, n_channels, height, width]

        if not isinstance(crop_batches, list):
            crop_batches = [crop_batches]
        
        # encode batch for every crop and merge
        # -> [n_crops, n_batches, embed_dim]
        embeddings = map(self.enc, crop_batches) 
        embeddings = torch.stack(embeddings, dim=0)     

        # compute outputs from embeddings
        # -> [n_crops, n_batches, out_dim]
        y = self.head(embeddings)
        return y




mc_spec = [
        {'name':'global1', 'out_size':244, 'min_scale':0.4, 'max_scale':1, 'teacher':True, 'student':True},
        {'name':'global2', 'out_size':244, 'min_scale':0.4, 'max_scale':1, 'teacher':True, 'student':True},
    ]
class MultiCropAugmentation(torch.nn.Module):
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
                    scale = (crop_spec['min_scale'], crop_spec['max_scale']),
                    interpolation = Image.BILINEAR
                )

    def forward(self, img):    
        crops = []
        for transform in self.transforms:
            crop = transform(img)
            crop = self.per_crop_transform(crop)
            crops.append(crop)

        return crops


class DINOTeacherUpdate(pl.Callback):
    def __init__(self,
        mode: str = 'ema',
        mom: float= 0.996, # Typing? Scheduler
        ) -> None:
        super().__init__()

        if mode == 'ema':
            self.mom = nn.Parameter(mom, requires_grad=False)
            self.on_batch_end = self.ema

        elif mode == 'prev_epoch':
            self.mom = nn.Parameter(0, requires_grad=False)
            self.on_epoch_end = self.ema

        elif mode == 'sgd':
            self.lr = nn.Parameter(1 - mom, requires_grad=False)
            self.wd = nn.Parameter(1 - mom, requires_grad=False)

            self.opt = torch.optim.SGD(lr=self.lr, weigh_decay=self.wd)
            self.on_batch_end = self.opt_step
            
        else:
            raise RuntimeError('Unkown teacher update mode.')

    def ema(self, dino: pl.LightningModule):
        for p_s, p_t in zip(dino.student.parameters(), dino.teacher.parameters()):
            p_t.data = self.m * p_t.data + (1 - self.m) * p_s.data

    def opt_step(self, dino: pl.LightningModule):
        with torch.no_grad:
            w_s = U.module_to_vector(dino.student)
        w_t = U.module_to_vector(dino.teacher)

        # optimizing similarity with sgd should be equal to ema
        loss = torch.dot(w_s, w_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class DINO(pl.LightningModule):
    def __init__(self,
        mc:MultiCropAugmentation,
        model:DINOModel,
        t_mode:str = 'ema',
        t_mom:Schedule = CosSched(0.996, 1),
        t_cmom:Schedule = ConstSched(0.9),
        t_temp:Schedule = LinWarmup(0.04, 0.04, 0),
        s_temp:Schedule = ConstSched(0.1),
        opt:Type[optim.Optimizer] = optim.Adam,
        opt_lr:Schedule = None,
        opt_wd:Schedule = None,
    ):
        super().__init__()
        self.embed_dim = model.embed_dim
        self.out_dim = model.out_dim
        
        # initiallize student and teacher with same params
        model_spec = type(model)
        self.student = model_spec()
        self.teacher = model_spec()
        self.student.load_state_dict(model.state_dict())
        self.teacher.load_state_dict(model.state_dict())
      
        # prepare teacher in evaluation mode
        self.teacher.eval() # TODO: will this stay in eval mode?
        for p in self.teacher.parameters():
            p.requires_grad = False

        # store crops
        for idx, crop in enumerate(mc.spec):
            if crop['teacher']:
                self.teacher.crops['name'] = crop['name']
                self.teacher.crops['idx'] = idx
            if crop['student']:
                self.student.crops['name'] = crop['name']
                self.student.crops['idx'] = idx

        # prepare schedulers for optimization procedure
        self.scheduler = Scheduler()

        # configure teacher updater
        self.t_updater = DINOTeacherUpdate(t_mode)
        self.scheduler.add(self.t_updater, 'mom', t_mom)
        
        # configure teacher temperature and centering
        self.scheduler.add(self.teacher.head, 'cmom', t_cmom)
        self.scheduler.add(self.teacher.head, 'temp', t_temp)
        self.scheduler.add(self.student.head, 'temp', s_temp)

        # configure optimizer, learning rate & weight decay
        params = self.student.named_parameters()
        self.optimizer = opt([
            {'params':[p for n,p in params if U.is_bias(n,p)]},
            {'params':[p for n,p in params if not U.is_bias(n,p)]}])

        if opt_lr is not None:
            self.scheduler.add(self.optimizer.param_groups[0], 'lr', opt_lr)
            self.scheduler.add(self.optimizer.param_groups[1], 'lr', opt_lr)
        if opt_wd is not None: # no weight decay for bias parameters
            self.scheduler.add(self.optimizer.param_groups[1], 'weight_decay', opt_wd)


    def configure_callbacks(self):
        return [self.scheduler, self.t_updater]

    def configure_optimizers(self):
        return self.optimizer


    def multicrop_loss(self, log_preds: torch.Tensor, log_targs: torch.Tensor):
        # [n_crops, n_batches, out_dim]

        # compute softmax outputs
        preds = torch.exp(log_preds)
        targs = torch.exp(log_targs)

        # compute per-crop entropy
        H_preds = U.entropy(preds, log_preds) # [n_crops, n_batches]
        H_targs = U.entropy(targs, log_targs) # [n_crops, n_batches]

        # compute pairwise losses
        KLs, CEs = [], []
        for i_stud, targ, H_targ in zip(self.teacher.crops['idx'], targs, H_targs):  
            for i_teach, log_pred in zip(self.student.crops['idx'], log_preds):
                
                # don't match the same views
                if i_stud == i_teach: 
                    continue

                CEs.append(U.cross_entropy(log_pred, targ))         # [n_batches]
                KLs.append(CEs[-1] - H_targ) # U.kl_divergence()    # [n_batches]

        # gather losses
        out = {}     
        out['CE'] = torch.mean(CEs)  
        out['KL'] = torch.mean(KLs)

        out['H_preds'] = {'mean':H_preds.mean()}
        for (crop_name,_), H_pred in zip(self.student.crops['name'], H_preds):
            out['H_preds'][crop_name] = H_pred.mean()

        out['H_targs'] = {'mean':H_targs.mean()}
        for (crop_name,_), H_targ in zip(self.teacher.crops['name'], H_targs):
            out['H_targs'][crop_name] = H_targ.mean()
        return out
 
    def training_step(self, batch: torch.Tensor):

        # generate teacher's targets and student's predictions
        # [n_crops, n_batches, n_channels, height, width]
        # -> [n_crops, n_batches, out_dim]
        with torch.no_grad:
            y_teacher_log = self.teacher(batch[self.teacher.crops['idx']])
        y_student_log = self.student(batch[self.student.crops['idx']])
        
        # compute multicrop loss
        out = self.multicrop_loss(y_student_log, y_teacher_log)

        ## logging
        self.log_dict(out)
        return out['CE']

    def validation_step(self, batch: torch.Tensor):
    
        # generate teacher's targets and student's predictions
        # [n_crops, n_batches, n_channels, height, width]
        # -> [n_crops, n_batches, out_dim]
        y_teacher_log = self.teacher(batch[self.teacher.crops['idx']])
        y_student_log = self.student(batch[self.student.crops['idx']])
        
        # compute multicrop loss
        out = self.multicrop_loss(y_student_log, y_teacher_log)

        ## logging
        out = dict((f'val_{k}', v) for (k,v) in out.items())
        self.log_dict(out)
        return out['CE']
