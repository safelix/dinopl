import torch
import torchvision
from .autoaugment import CIFAR10Policy
from .tinyimagenet import TinyImageNet


def get_transforms(augmentation, size=32):
    if augmentation == "none":
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )
    elif augmentation == "full":
        color_jitter = torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=32),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )
    elif augmentation == "c10":
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(size=32, padding=4),
                CIFAR10Policy(),
                torchvision.transforms.ToTensor(),
            ]
        )

        raise NotImplementedError("TODO")

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=size),
            torchvision.transforms.ToTensor(),
        ]
    )

    return train_transform, test_transform


def get_dataloaders(dataset, augmentation, size=32):
    train_transform, test_transform = get_transforms(augmentation, size=size)

    dataset = dataset.lower()

    if dataset == "cifar-10":
        train_dataset = torchvision.datasets.CIFAR10(
            "./datasets/", download=True, transform=train_transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            "./datasets/", download=True, transform=test_transform, train=False
        )
    elif dataset == "cifar-100":
        train_dataset = torchvision.datasets.CIFAR100(
            "./datasets/", download=True, transform=train_transform
        )

        test_dataset = torchvision.datasets.CIFAR100(
            "./datasets/", download=True, transform=test_transform, train=False
        )
    elif dataset == "stl-10":
        train_dataset = torchvision.datasets.STL10(
            "./datasets/", split="train", download=True, transform=train_transform
        )

        test_dataset = torchvision.datasets.STL10(
            "./datasets/", split="test", download=True, transform=test_transform
        )
    elif dataset == "tiny-imagenet":
        train_dataset = TinyImageNet(
            "./datasets/", download=True, split="train", transform=train_transform
        )

        test_dataset = TinyImageNet(
            "./datasets/", download=True, split="val", transform=test_transform
        )
    else:
        raise NotImplementedError("TODO")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=6
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=6
    )

    return train_loader, test_loader


def get_optimizer(model, name):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=1e-2)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        raise NotImplementedError("TODO")
