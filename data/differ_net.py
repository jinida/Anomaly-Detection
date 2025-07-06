from data import MVTEC_MEAN_STD
from data.base import MVTecBase

from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms.functional import rotate

class MVTecMultiTransform(MVTecBase):
    def __init__(self, 
                root_dir: str, 
                category: str,
                is_train: bool = True,
                image_size: tuple = (448, 448),
                n_transforms: int = 4,
                fixed_rotations: bool = False,
                repeat=10):
        super().__init__(root_dir, categories=[category], is_train=is_train)

        self.n_transforms = n_transforms
        self.fixed_rotation_mode = fixed_rotations
        self.repeat = repeat
        self.image_size = image_size
        self.category = category.lower()
        self.random_transform = self._build_random_transform()
        self.fixed_degrees = [i * 360.0 / n_transforms for i in range(n_transforms)]
            
    def _build_random_transform(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MVTEC_MEAN_STD[self.category][0],
                                 std=MVTEC_MEAN_STD[self.category][1])
        ])
    
    def _get_fixed_rotation_transform(self, degrees):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            lambda x: rotate(x, degrees, resample=False, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=MVTEC_MEAN_STD[self.category][0],
                                 std=MVTEC_MEAN_STD[self.category][1])
        ])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        index = index % len(self.image_paths)
        path = self.image_paths[index]
        target = self.labels[index]
    
        sample = Image.open(path).convert('RGB')

        samples = []
        for i in range(self.n_transforms):
            if self.fixed_rotation_mode:
                degrees = self.fixed_degrees[i]
                transform = self._get_fixed_rotation_transform(degrees)
                samples.append(transform(sample))
            else:
                samples.append(self.random_transform(sample))
        
        return torch.stack(samples, dim=0), target
    
    def __len__(self) -> int:
        return len(self.image_paths) * self.repeat