import torch
from torch import device, nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision import models
from torchvision import transforms
from configuration import CONSTANTS as C, create_optimizer
from configuration import Configuration, create_encoder

from torch.nn import functional as F


self_trfm = transforms.Compose([ # self-training
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.ToTensor()])
eval_trfm = transforms.Compose([ # evaluation
                transforms.Resize(size=128),
                self_trfm])
                

config = Configuration.get_default()

# Data Loading
config.n_classes = 10
eval_train_set = MNIST(root=C.DATA_DIR, train=True, transform=eval_trfm)
eval_train_dl = DataLoader(dataset=eval_train_set, batch_size=config.bs_eval)

# Training

encoder = create_encoder(config).to(C.DEVICE)

@torch.enable_grad()
def train_linear(embed_dim, n_classes, enc, dl, dev):
    clf = nn.Linear(embed_dim, n_classes, device=dev)
    opt = torch.optim.Adam(clf.parameters())

    for batch in dl:
        inputs = batch[0].to(dev)
        targets = batch[1].to(dev)

        #inputs.requires_grad = True
        #targets.requires_grad = True
        print('\ninp:', inputs.shape, inputs.requires_grad)
        print('targ:', targets.shape, targets.requires_grad)

        # get embeddings
        with torch.no_grad():
            embeddings = enc(inputs)
        print('emb:', embeddings.shape, embeddings.requires_grad)

        # get predictions
        opt.zero_grad()
        predictions = clf(embeddings)
        print('pred:', predictions.shape, predictions.requires_grad)

        loss = F.cross_entropy(predictions, targets)
        print('loss:', loss, loss.requires_grad)

        loss.backward()
        opt.step()
        break

    return clf

@torch.no_grad()
def valid_linear(clf, enc, dl, dev):
    accuracy = Accuracy()
    
    with torch.no_grad():
        for batch in dl:
            inputs = batch[0].to(dev)
            targets = batch[1].to(dev)

            # compute predictions
            predictions = clf(enc(inputs))
            accuracy.update(predictions, targets)

    return float(accuracy.compute())


#train_linear(config.embed_dim, config.n_classes, C.DEVICE, encoder, eval_train_dl)