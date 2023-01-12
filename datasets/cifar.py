import os
from typing import Callable, Optional
import numpy as np
import torchvision.datasets as datasets

__all__ = ["CIFAR10", "CIFAR100"]


class CIFAR10(datasets.CIFAR10):
    ds_pixels = 1024
    ds_channels = 3
    ds_classes = 10
    mean = (
        0.4914,
        0.4822,
        0.4465,
    )  # = (torch.tensor(self.data) / 255).mean(dim=[0,1,2])
    std = (0.247, 0.243, 0.261)  # = (torch.tensor(self.data) / 255).std(dim=[0,1,2])

    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dataset_size: int = None,
        **kwargs,
    ) -> None:
        defaults = dict(root=os.environ["DINO_DATA"], download=True)
        defaults.update(kwargs)

        super().__init__(
            train=train,
            transform=transform,
            target_transform=target_transform,
            **defaults,
        )

        # assume that we filter the dataset only for the training split
        if train and dataset_size is not None and dataset_size < 50000:
            percentage_per_class_to_keep = dataset_size / 50000

            # filter dataset in a balanced way
            classes = list(range(len(self.classes)))
            indices_per_class = {k: [] for k in classes}

            for i, t in enumerate(self.targets):
                if t in classes:
                    indices_per_class[t].append(i)

            for i in classes:
                np.random.shuffle(indices_per_class[i])

                indices_per_class[i] = indices_per_class[i][
                    : int(len(indices_per_class[i]) * percentage_per_class_to_keep)
                ]

            indices = np.concatenate(list(indices_per_class.values()))
            np.random.shuffle(indices)

            self.data = np.array([self.data[i] for i in indices])
            self.targets = np.array([self.targets[i] for i in indices])
        elif train and dataset_size is not None and dataset_size > 50000:
            # load cifar10 and load from cifar-5m to complete the requested

            data_x = [self.data]
            data_y = [np.array(self.targets)]

            total_size = dataset_size - 50000

            for i in range(6):
                part_path = os.path.join(
                    os.environ["DINO_DATA"], "cifar-5m", f"part{i}.npz"
                )
                print("Loading", part_path)
                data = np.load(part_path)

                data_x.append(data["X"])
                data_y.append(data["Y"])

                total_size -= data["X"].shape[0]

                if total_size <= 0:
                    break

            self.data = np.concatenate(data_x, axis=0)[: int(dataset_size)]
            self.targets = np.concatenate(data_y, axis=0)[: int(dataset_size)]

        print("Loaded CIFAR10 train: {} with {} samples".format(train, len(self)))


class CIFAR100(datasets.CIFAR100):
    ds_pixels = 1024
    ds_channels = 3
    ds_classes = 10
    mean = (
        0.4914,
        0.4822,
        0.4465,
    )  # = (torch.tensor(self.data) / 255).mean(dim=[0,1,2])
    std = (0.247, 0.243, 0.261)  # = (torch.tensor(self.data) / 255).std(dim=[0,1,2])

    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        defaults = dict(root=os.environ["DINO_DATA"], download=True)
        defaults.update(kwargs)
        super().__init__(
            train=train,
            transform=transform,
            target_transform=target_transform,
            **defaults,
        )


if __name__ == "__main__":
    path_to_data = os.environ["DINO_DATA"]
    os.makedirs(path_to_data, exist_ok=True)
    CIFAR10(root=path_to_data, download=True)
    pass
