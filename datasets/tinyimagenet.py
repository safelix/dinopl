"""Simple Tiny ImageNet dataset utility class for pytorch."""

import os

import shutil

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive

from .base import BaseDataset

__all__ = ['TinyImageNet']


def normalize_tin_val_folder_structure(
    path, images_folder="images", annotations_file="val_annotations.txt"
):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError("Validation folder is empty.")
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()


class TinyImageNetFolder(ImageFolder):
    """Dataset for TinyImageNet-200"""

    base_folder = "tiny-imagenet-200"
    zip_md5 = "90528d7ca1a48142e341f4ef8d21d0de"
    splits = ("train", "val")
    filename = "tiny-imagenet-200.zip"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(self, root:str, train: bool = True, download:bool = False, **kwargs):
        self.data_root = os.path.expanduser(root)

        split = 'train' if train else 'val'
        self.split = verify_str_arg(split, "split", self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        super().__init__(self.split_folder, **kwargs)

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def download(self):
        if self._check_exists():
            print("Files already downloaded and verified")
            return
        
        download_and_extract_archive(
            self.url,
            self.data_root,
            filename=self.filename,
            remove_finished=True,
            md5=self.zip_md5,
        )

        assert "val" in self.splits
        normalize_tin_val_folder_structure(os.path.join(self.dataset_folder, "val"))


class TinyImageNet(BaseDataset, TinyImageNetFolder):
    img_size = (64, 64)
    ds_pixels = 4096
    ds_channels = 3
    ds_classes = 1000
    mean = (0.485, 0.456, 0.406)    # from ImageNet
    std = (0.229, 0.224, 0.225)     # from ImageNet

# run with 'python -m datasets.tinyimagenet'
if __name__ == '__main__':
    path_to_data = os.environ['DINO_DATA']
    os.makedirs(path_to_data, exist_ok=True) 
    print(TinyImageNet(root=path_to_data, download=True))
