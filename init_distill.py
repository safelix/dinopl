import copy
import os

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from models import resnet, convnet, vgg
from torchvision import models as tvm


import dinopl.utils as U
from dinopl.probing import LinearProbe, LinearProber

torch.backends.cudnn.benchmark = True

logfile = 'results/logs_resnet_fanin_nobn_sgd.csv'
#logfile_lr_coeffs = 'results/lr_coeffs.csv'
#logfile_grad_old_coeffs = 'results/grad_old_coeffs.csv'
#logfile_grad_new_coeffs = 'results/grad_new_coeffs.csv'
#logfile_steps_created = False

n_epochs = 100
batch_size = 256
device = torch.device(f'cuda:{U.pick_single_gpu()}')

l2bot_dim = 256
out_dim = 2**16

t_temp, s_temp = 1, 1

# Prepare data.
trfm = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')), transforms.ToTensor(), 
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
train_set = CIFAR10(root=os.environ['DINO_DATA'], train=True, transform=trfm)
valid_set = CIFAR10(root=os.environ['DINO_DATA'], train=False, transform=trfm)
train_loader = DataLoader(train_set, batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
probe_train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
probe_eval_loader = DataLoader(valid_set, batch_size, num_workers=4, pin_memory=True)

# Prepare resnet encoder.
#enc = convnet.convnet(width=16, depth=3, norm_layer=nn.BatchNorm2d)

#enc = tvm.vgg11()
#enc = vgg.vgg11()
#enc = vgg.vgg11(norm_layer=(lambda dim: nn.BatchNorm2d(dim, affine=True))) # BatchNorm
#enc = vgg.vgg11(norm_layer=(lambda dim: nn.GroupNorm(1, dim, affine=True))) # LayerNorm
#enc = vgg.vgg11(norm_layer=(lambda dim: nn.GroupNorm(dim, dim, affine=True))) # InstanceNorm
#enc.classifier = nn.Identity()
#enc.avgpool = nn.Identity()

#enc = tvm.resnet18()
#enc = resnet.resnet18()
enc = resnet.resnet18(preact=False, num_classes=None, norm_layer=nn.Identity, tiny_input=True)
#enc = resnet.resnet18(preact=False, norm_layer=(lambda dim: norm.BatchNorm2d(dim, compute_mean=False, compute_var=True, affine=True)))
#enc = resnet.resnet18(norm_layer=(lambda dim: nn.BatchNorm2d(dim, track_running_stats=True, affine=True)))
#enc = resnet.resnet18(norm_layer=(lambda dim: nn.GroupNorm(1, dim, affine=True))) # LayerNorm
#enc = resnet.resnet18(norm_layer=(lambda dim: nn.GroupNorm(dim, dim, affine=True))) # InstanceNorm
#enc = nf_resnet.nf_resnet18(base_conv=nn.Conv2d)

#for m in enc.modules():
#    if isinstance(m, nn.Conv2d):
#        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#        #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
#nn.init.kaiming_normal_(enc.conv1.weight, mode='fan_in', nonlinearity='linear')
enc.reset_parameters(mode='fan_in', nonlinearity='linear')
embed_dim = enc.embed_dim

# Prepare mlp projection head.
head = nn.Sequential(nn.Linear(embed_dim, 2048), nn.ReLU(),
                        nn.Linear(2048, 2048) , nn.ReLU())

for m in head.modules(): # initialize head weights
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Create student and teacher.
student = nn.Module()
student.add_module('enc', enc)
student.add_module('head', head)
student.add_module('bot_layer', nn.Linear(2048, 256, bias=True))
student.add_module('wn_layer', nn.Linear(256, out_dim, bias=False))
student.to(device)



teacher = copy.deepcopy(student)

# Create optimizer.
student.requires_grad_(True)
teacher.requires_grad_(False)
params = list(student.named_parameters()) # generator -> list
opt = SGD([
    {'params':[p for n,p in params if not U.is_bias(n,p)], 'weight_decay':1e-4},
    {'params':[p for n,p in params if U.is_bias(n,p)], 'weight_decay':0}], lr=1e-3, momentum=0.9)

opt = AdamW([
    {'params':[p for n,p in params if not U.is_bias(n,p)], 'weight_decay':1e-2},
    {'params':[p for n,p in params if U.is_bias(n,p)], 'weight_decay':0}], lr=1e-3)

opt = AdamW(student.parameters(), lr=1e-2, weight_decay=0) # fullbatch gradient descent

opt = AdamW(student.enc.parameters(), lr=1e-3, weight_decay=0) # frozen head
opt = AdamW(student.parameters(), lr=1e-3, weight_decay=0)

opt = AdamW(student.parameters(), lr=1e-3, weight_decay=0, betas=(0, 0)) # resnet nobn
opt = SGD(student.parameters(), lr=1e-3)

# Create LinearProber
probes = {}
#probes['teacher'] = LinearProbe(teacher.enc, embed_dim, 10)
probes['student'] = LinearProbe(student.enc, embed_dim, 10)
prober = LinearProber(-1, 10, probes, probe_train_loader, probe_eval_loader, seed=1234567890)

# Calibrate batchnorms.
#teacher.train(), student.train()
#progress_bar = tqdm(train_loader, desc=f'Calibration')
#for x, _ in progress_bar:
#    student.enc(x.to(device))
#    teacher.enc(x.to(device))

#for m in student.modules():
#    if isinstance(m, nn.BatchNorm2d):
#        m.running_mean = None
#        m.running_var = None

# Training.
teacher.eval(), student.eval()

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
            #loss = loss / len(train_loader) # fullbatch gradient descent
            loss.backward()


        # Freeze batchnorm params.
        #for m_s in student.modules():
        #    if isinstance(m_s, torch.nn.modules.batchnorm._BatchNorm):
        #        for p in m_s.parameters(recurse=False):
        #            p.grad = None

        #lr_coeffs = pd.DataFrame()
        #grad_old_coeffs = pd.DataFrame()
        #grad_new_coeffs = pd.DataFrame()
        for n, p in student.named_parameters():
            #grad_old_coeffs[f'{n}.mean'] = [p.grad.abs().mean().item()]
            #grad_old_coeffs[f'{n}.std'] = [p.grad.abs().std().item()]

            coeffs = ((p.grad*p.grad).sqrt() + 1e-8)
            #lr_coeffs[f'{n}.mean'] = [coeffs.mean().item()]
            #lr_coeffs[f'{n}.std'] = [coeffs.std().item()]

            p.grad /= coeffs
            #grad_new_coeffs[f'{n}.mean'] = [p.grad.abs().mean().item()]
            #grad_new_coeffs[f'{n}.std'] = [p.grad.abs().std().item()]

        #if not logfile_steps_created:
        #    lr_coeffs.to_csv(logfile_lr_coeffs, index=False)
        #    grad_old_coeffs.to_csv(logfile_grad_old_coeffs, index=False)
        #    grad_new_coeffs.to_csv(logfile_grad_new_coeffs, index=False)
        #    logfile_steps_created = True
        #else:
        #    lr_coeffs.to_csv(logfile_lr_coeffs, header=False, index=False, mode='a')
        #    grad_old_coeffs.to_csv(logfile_grad_old_coeffs, header=False, index=False, mode='a')
        #    grad_new_coeffs.to_csv(logfile_grad_new_coeffs, header=False, index=False, mode='a')
            


        opt.step() # minibatch gradient descent
        student.zero_grad(True) # minibatch gradient descent

        # log to progress bar
        progress_bar.set_postfix({'loss':loss.item(), 'norm':U.module_to_vector(student.enc).norm(p=2).item()})


    #opt.step() # fullbatch gradient descent
    #student.zero_grad(True) # fullbatch gradient descent
    logs.to_csv(logfile)