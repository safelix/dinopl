import torch
from torch import nn
from typing import List, Any, Tuple
from torchvision import transforms
from torchvision.datasets import VisionDataset


mc_spec = [
        {'name':'global1', 'out_size':244, 'min_scale':0.4, 'max_scale':1, 'teacher':True, 'student':True},
        {'name':'global2', 'out_size':244, 'min_scale':0.4, 'max_scale':1, 'teacher':True, 'student':True},
    ]
class MultiCrop(nn.Module):
    f'''
    Takes a list of crop specifications.
    Example:
    {mc_spec}
    '''

    def __init__(self, spec:List[dict], per_crop_transform=nn.Identity):
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
                    scale = (crop_spec['min_scale'], crop_spec['max_scale'])
                )

    def forward(self, img):
        # [height, width, (n_channels)]
        # -> [n_crops, n_channels, height, width]
        crops = []
        for name, transform in self.transforms.items():
            crop = transform(img)
            crop = self.per_crop_transform(crop)
            crops.append(crop)
        return crops


class LabelNoiseWrapper(VisionDataset):
    def __init__(self, dataset, n_classes, noise_ratio=1, resample=False) -> None:
        super().__init__(root=None)
        if noise_ratio < 0 or 1 < noise_ratio:
            raise RuntimeError(f'Noise ratio \'{noise_ratio}\' not allowed. Needs to be in [0,1].')
        self.dataset = dataset
        self.n_classes = n_classes
        self.noise_ratio = noise_ratio
        self.resample = resample
        if resample:  # no precomputation of targets
            return

        # get true target targets as defaults
        self.targets = torch.zeros((len(self.dataset),), dtype=int)
        for index, (img, target) in enumerate(self.dataset):
            self.targets[index] = target

        # select indices of noisy targets and overwrite with new target
        indices = torch.randperm(len(self.dataset))[:round(len(self.dataset) * noise_ratio)]
        self.targets[indices] = torch.randint_like(indices, self.n_classes)
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.dataset.__getitem__(index)

        if not self.resample: # return precomputed noise target
            return img, self.targets[index]
        
        # overwrite original target with propability self.noise_ratio
        if torch.rand((1,)) < self.noise_ratio:
            target = torch.randint(0, self.n_classes, (1,))
        return img, target

    def __len__(self) -> int:
        return self.dataset.__len__()

    def extra_repr(self) -> str:
        body = [f'Wrapping: {self.dataset.__repr__()}']
        body += [f'n_classes: {self.n_classes}']
        body += [f'noise_ratio: {self.noise_ratio}']
        body += [f'resample: {self.resample}']
        return '\n'.join(body)

    

class LogitNoiseWrapper(VisionDataset):
    def __init__(self, dataset, n_classes, temperature=1, resample=False) -> None:
        super().__init__(root=None)
        if temperature <= 0:
            raise RuntimeError(f'Temperature \'{temperature}\' not allowed. Needs to be >0.')
        self.dataset = dataset
        self.n_classes = n_classes
        self.temperature = temperature
        self.resample = resample
        if resample:  # no precomputation of targets
            return

        # precompute noisy logits from normal distribution 
        self.targets = torch.randn((len(self.dataset), self.n_classes)) 
        self.targets = self.targets / sqrt(self.n_classes) # scale to l2-norm of 1
       
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.dataset.__getitem__(index)

        if not self.resample: # return precomputed noise target
            return img, self.targets[index] / self.temperature
        
        target = torch.randn((1, self.n_classes)) / self.temperature
        return img, target

    def __len__(self) -> int:
        return self.dataset.__len__()

    def extra_repr(self) -> str:
        body = [f'Wrapping: {self.dataset.__repr__()}']
        body += [f'n_classes: {self.n_classes}']
        body += [f'temperature: {self.temperature}']
        body += [f'resample: {self.resample}']
        return '\n'.join(body)
        