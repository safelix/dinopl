import time
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from configuration import CONSTANTS as C, create_optimizer
from configuration import Configuration, create_encoder

from dino import *
from eval_linear import train
from probing import LinearProbe, ProbingCallback

import my_utils as U

def main(config:Configuration):
    # Fix random seed
    if config.seed is None:
        config.seed = int(time.time())

    # Multi-Crop Augmentation
    MC_SPEC = [
            {'name':'global1', 'out_size':128, 'min_scale':0.6, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':128, 'min_scale':0.6, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'local1', 'out_size':96, 'min_scale':0.2, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local2', 'out_size':96, 'min_scale':0.2, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local3', 'out_size':96, 'min_scale':0.2, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local4', 'out_size':96, 'min_scale':0.2, 'max_scale':0.4, 'teacher':False, 'student':True},
        ]

    
    self_trfm = transforms.Compose([ # self-training
                    transforms.Lambda(lambda img: img.convert('RGB')),
                    transforms.ToTensor()])
    eval_trfm = transforms.Compose([ # evaluation
                    transforms.Resize(size=128),
                    self_trfm])
                    
    mc = MultiCropAugmentation(MC_SPEC, per_crop_transform=self_trfm)

    # Data Loading
    self_train_set = MNIST(root=C.DATA_DIR, train=True, transform=mc)
    self_valid_set = MNIST(root=C.DATA_DIR, train=False, transform=mc)
    eval_train_set = MNIST(root=C.DATA_DIR, train=True, transform=eval_trfm)
    eval_valid_set = MNIST(root=C.DATA_DIR, train=False, transform=eval_trfm)
    config.n_classes = 10

    self_train_dl = DataLoader(dataset=self_train_set, batch_size=config.bs_train)
    self_valid_dl = DataLoader(dataset=self_valid_set, batch_size=config.bs_train)
    eval_train_dl = DataLoader(dataset=eval_train_set, batch_size=config.bs_eval)
    eval_valid_dl = DataLoader(dataset=eval_valid_set, batch_size=config.bs_eval)
   
    # Model Setup
    enc = create_encoder(config)
    head = DINOHead(config.embed_dim, config.out_dim, 
            hidden_dims=config.hid_dims, 
            l2_bottleneck_dim=config.bot_dim, 
            use_bn=config.use_bn,
            act_fn=config.mlp_act)
    model = DINOModel(enc, head)
    #print(model)


    # DINO Setup
    dino = DINO(mc=mc, model=model,
                t_mode = config.t_mode,
                t_mom  = config.t_mom,
                t_cmom = config.t_cmom,
                t_temp = config.t_temp,
                s_temp = config.s_temp,
                opt = create_optimizer(config),
                opt_lr = config.opt_lr,
                opt_wd = config.opt_wd)


    # Tracking Logic
    probing_cb = ProbingCallback(
        probes={'Student': LinearProbe(dino.student.enc, 
                                        config.embed_dim, 
                                        config.n_classes),
                'Teacher': LinearProbe(dino.teacher.enc, 
                                        config.embed_dim, 
                                        config.n_classes)},
        train_dl=eval_train_dl,
        valid_dl=eval_valid_dl,
        probe_every=config.probe_every,
        probing_epochs=config.probing_epochs,
    )


    # Training
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator='auto',
        gradient_clip_val=config.clip_grad,
        callbacks=[probing_cb])
    trainer.fit(model=dino, 
                train_dataloaders=[self_train_dl],
                val_dataloaders=[self_valid_dl])


if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)
    main(config)
