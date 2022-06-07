import time
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from configuration import CONSTANTS as C, create_optimizer
from configuration import Configuration, create_encoder

from dino import *
from probing import LinearProbe, ProbingCallback

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
    mc = MultiCropAugmentation(MC_SPEC)

    # Data Loading
    self_train_set = MNIST(root=C.DATA_DIR, train=True, transform=mc)
    eval_train_set = MNIST(root=C.DATA_DIR, train=True)
    eval_valid_set = MNIST(root=C.DATA_DIR, train=False)

    self_train_dl = DataLoader(dataset=self_train_set, batch_size=config.bs_train)
    eval_train_dl = DataLoader(dataset=eval_train_set, batch_size=config.bs_eval)
    eval_valid_dl = DataLoader(dataset=eval_valid_set, batch_size=config.bs_eval)

    # Model Setup
    enc = create_encoder(config)
    head = DINOHead(config.embed_dim, config.out_dim, 
            hidden_dims=config.hid_dims, 
            bottleneck_dim=config.bot_dim, 
            use_bn=config.use_bn,
            act_fn=config.mlp_act)
    model = DINOModel(enc, head)

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
        probes={'Student': LinearProbe(config, dino.student.encoder),
                'Teacher': LinearProbe(config, dino.teacher.encoder)},
        train_dl=eval_train_dl,
        valid_dl=eval_valid_dl,
    )


    # Training
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator='auto',
        gradient_clip_val=config.clip_grad,
        callbacks=[probing_cb])
    trainer.fit(model=dino, train_dataloader=self_train_dl)


if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)
    main(config)
