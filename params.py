import torch
from torch import nn
from dino import DINOHead
from scheduling import Schedule, ConstSched, Record

lin = nn.Linear(1, 1)
opt = torch.optim.Adam(lin.parameters())

head = DINOHead(512, 256)


rec = Record(head.__dict__, 'temp', ConstSched(0.9))
