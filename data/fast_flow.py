import os

import numpy as np
from PIL import Image
import torch

from data.base import MVTecBase

class MVTecFF(MVTecBase):
    def __init__(self, root_dir: str, category: str, is_train: bool = True, image_size: tuple = (256, 256), repeat=50):
        self.image_size = image_size
        self.category = category.lower()
        super().__init__(root_dir, categories=[category], is_train=is_train)
        self.transform = self._build_transform()
        self.repeat = repeat
        
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        index = index % len(self.image_paths)
        image_file = self.image_paths[index]
        image = Image.open(image_file).convert('RGB')
        image = np.array(image)
        image = self.image_transform(image=image)['image']
        
        if self.is_train:
            return image
        else:
            target_file = self.mask_paths[index]
            if target_file and os.path.exists(target_file):
                target = Image.open(target_file).convert('L')
                target = np.array(target, dtype=np.float32) / 255.0
            else:
                target = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float32) / 255.
            target = self.target_transform(image=target)['image']
            return image, self.labels[index], target
        
    def __len__(self):
        return len(self.image_paths) * self.repeat
        

        