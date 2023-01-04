import copy
import os

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchvision import models as tvm
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import dinopl.utils as U
from dinopl.modules import init
from dinopl.probing import LinearAnalysis, Prober
from models import convnet, resnet, vgg

generator = torch.Generator().manual_seed(570880439925033212)
device = torch.device(f'cuda:{U.pick_single_gpu()}')
logfile = 'results/logs_vgg_bn.csv'


n_epochs = 100
batch_size = 256

l2bot_dim = 256
out_dim = 2**16

t_temp, s_temp = 1, 1


# Prepare data.
trfm = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')), transforms.ToTensor(), 
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
train_set = CIFAR10(root=os.environ['DINO_DATA'], train=True, transform=trfm)
valid_set = CIFAR10(root=os.environ['DINO_DATA'], train=False, transform=trfm)
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, generator=generator)
probe_train_loader = DataLoader(train_set, batch_size, shuffle=False, num_workers=4, pin_memory=True)
probe_eval_loader = DataLoader(valid_set, batch_size, num_workers=4, pin_memory=True)


# Prepare encoder.
#enc = convnet.convnet(width=16, depth=1.5, norm_layer=nn.BatchNorm2d)

enc = vgg.vgg11_bn(num_classes=None)
#enc = vgg.vgg11(num_classes=None, norm_layer=(lambda dim: nn.BatchNorm2d(dim, affine=True))) # BatchNorm
#enc = vgg.vgg11(num_classes=None, norm_layer=(lambda dim: nn.GroupNorm(1, dim, affine=True))) # LayerNorm
#enc = vgg.vgg11(num_classes=None, norm_layer=(lambda dim: nn.GroupNorm(dim, dim, affine=True))) # InstanceNorm

#enc = resnet.resnet18(num_classes=None, tiny_input=True)
#enc = resnet.resnet18(num_classes=None, tiny_input=True, norm_layer=nn.Identity)
#enc = resnet.resnet18(num_classes=None, tiny_input=True, norm_layer=(lambda dim: nn.BatchNorm2d(dim))  # BatchNorm
#enc = resnet.resnet18(num_classes=None, tiny_input=True, norm_layer=(lambda dim: nn.GroupNorm(1, dim))) # LayerNorm
#enc = resnet.resnet18(num_classes=None, tiny_input=True, norm_layer=(lambda dim: nn.GroupNorm(dim, dim))) # InstanceNorm

enc.reset_parameters(generator=generator)
embed_dim = enc.embed_dim

# Alternative: torchvision models
#enc = tvm.vgg11_bn()
#enc.avgpool = nn.Identity()
#enc.classifier = nn.Identity()

#enc = tvm.resnet18()
#enc = U.modify_resnet_for_tiny_input(enc)
#enc.fc = nn.Identity()


# Prepare mlp projection head.
head = nn.Sequential(nn.Linear(embed_dim, 2048), nn.ReLU(),
                        nn.Linear(2048, 2048) , nn.ReLU())

for m in head.modules(): # initialize head weights
    if isinstance(m, nn.Linear):
        init.trunc_normal_(m.weight, std=.02, generator=generator)
        if m.bias is not None:
            init.constant_(m.bias, 0)

# Prepare l2-bottleneck output layer
bot_layer = nn.Linear(2048, 256, bias=True)
wn_layer = nn.Linear(256, out_dim, bias=False)

init.trunc_normal_(bot_layer.weight, std=.02, generator=generator)
init.constant_(bot_layer.bias, 0)
init.trunc_normal_(wn_layer.weight, std=.02, generator=generator)
#nn.init.constant_(student.wn_layer.bias, 0)


# Create student and teacher.
student = nn.Module()
student.add_module('enc', enc)
student.add_module('head', head)
student.add_module('bot_layer', bot_layer)
student.add_module('wn_layer', wn_layer)

student.to(device)
teacher = copy.deepcopy(student)


# Create optimizer.
student.requires_grad_(True)
teacher.requires_grad_(False)

opt = AdamW(student.parameters(), lr=1e-3, weight_decay=0)                  # standard Adam
opt = SGD(student.parameters(), lr=1e-3)                                    # standard SGD
opt = AdamW(student.parameters(), lr=1e-3, weight_decay=0, betas=(0, 0))    # no betas, pure adaptivity


# Calibrate batchnorms.
#teacher.train(), student.train()
#progress_bar = tqdm(train_loader, desc=f'Calibration')
#for x, _ in progress_bar:
#    student.enc(x.to(device))
#    teacher.enc(x.to(device))


# Create Prober
encoders = {'student':student.enc}#, 'teacher':teacher.enc} 
analyses = {'':LinearAnalysis(n_epochs=10)}
prober = Prober(encoders, analyses, probe_train_loader, probe_eval_loader, 10, normalize=False, seed=1234567890)


# Training.
teacher.eval(), student.train()

logs = pd.DataFrame()
print(f'Starting Training on {device}')
for epoch in range(n_epochs):
    probe = prober.probe(device) # perform linear probe
    logs = pd.concat([logs, pd.DataFrame(probe, index=[0])])

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for x, _ in progress_bar:
        x = x.to(device)

        with torch.no_grad():
            x_t = teacher.head(teacher.enc(x))

            # transform to bottleneck
            x_t = teacher.bot_layer(x_t)
            x_t = F.normalize(x_t, p=2, dim=-1)

            # prepare wn_layer
            V_t = teacher.wn_layer.weight
            V_t = F.normalize(V_t, p=2, dim=-1)
            
            # transform to logits
            x_t = F.linear(x_t, V_t)
            #x_t = F.normalize(x_t, p=2, dim=-1)

            # transform to targets
            target = F.softmax(x_t / t_temp, dim=-1)

        with torch.enable_grad():
            x_s = student.head(student.enc(x))

            # transform to bottleneck
            x_s = student.bot_layer(x_s)
            x_s = F.normalize(x_s, p=2, dim=-1)

            # prepare wn_layer
            V_s = student.wn_layer.weight
            V_s = F.normalize(V_s, p=2, dim=-1)

            # transform to logits
            x_s = F.linear(x_s, V_s)
            #x_s = F.normalize(x_s, p=2, dim=-1)

            # transform to predictions
            log_prediction = F.log_softmax(x_s / s_temp, dim=-1)

            # compute loss, backpropagate, update
            loss = torch.sum(target * -log_prediction, dim=-1).mean()
            #loss = (x_s - x_t).square().mean()
            loss.backward()

        # Manual adaptivity if sgd is used.
        #for n, p in student.named_parameters():
        #    coeffs = ((p.grad*p.grad).sqrt() + 1e-8)
        #    p.grad /= coeffs

        # Freeze mlp layers.
        #student.head.zero_grad(True)
        #student.bot_layer.zero_grad(True)
        #student.wn_layer.zero_grad(True)

        opt.step() # minibatch gradient descent
        student.zero_grad(True) # minibatch gradient descent

        # log to progress bar
        progress_bar.set_postfix({'loss':loss.item(), 'norm':U.module_to_vector(student.enc).norm(p=2).item()})

    logs.to_csv(logfile)