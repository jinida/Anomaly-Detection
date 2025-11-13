import os

import torch

from data.base import MVTecCategory, MVTecSingle
import albumentations as A

import numpy as np
from PIL import Image

class MVTecPC(MVTecSingle):
    def __init__(self, root_dir: str, category: str, is_train: bool = True, image_size: int = 256, crop_size: int = 224):
        self.image_size = (image_size, image_size)
        self.crop_size = (crop_size, crop_size)
        super(MVTecPC, self).__init__(root_dir, category, is_train)
        self._build_transform()
    
    def _build_transform(self):
        common_transform = [
            A.Resize(self.image_size[0], self.image_size[1]),
            A.CenterCrop(self.crop_size[0], self.crop_size[1])
        ]
        
        self.image_transform = A.Compose(
            common_transform + [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2()
            ]
        )
        self.target_transform = A.Compose(
            common_transform + [
                A.pytorch.ToTensorV2()
            ]
        )

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        
        image = np.array(Image.open(image_path).convert('RGB'))
        
        if mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        augmented = self.image_transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        mask = (mask > 0.5).float()

        return image, label, mask