import os

from typing import Union, Tuple
from functools import lru_cache

from PIL import Image
import numpy as np
import torch

from data.base import MVTecSingle
from data.utils import MVTecCategory, generate_coords_position, generate_coords_svdd
from data import MVTEC_MEAN_STD

import albumentations as A
from albumentations.pytorch import ToTensorV2

class PatchSSLDataset(MVTecSingle):
    def __init__(self,
                 root_path: str,
                 category: Union[MVTecCategory, str],
                 image_size: int = 256,
                 repeat: int = 100,
                 cache_size: int = 16):
        
        self.repeat = repeat
        self.image_size = image_size
        super().__init__(root_path, category=category, is_train=True)
        self._build_transform()
        self._get_image = lru_cache(maxsize=cache_size)(self._get_image)

    def _get_image(self, index: int) -> np.ndarray:
        image_path = self.image_paths[index]
        return np.array(Image.open(image_path).convert('RGB'))
        
    def __getitem__(self, idx):
        image_idx = idx % self.total_images
        image = self._get_image(image_idx)
        
        patch_size_64 = self.image_size // 4
        patch_size_32 = self.image_size // 8

        p1, p2, pos64 = generate_coords_position(self.image_size, self.image_size, K=patch_size_64)
        patch1_p64 = image[p1[0]:p1[0]+patch_size_64, p1[1]:p1[1]+patch_size_64].copy().astype(np.float32)
        patch2_p64 = image[p2[0]:p2[0]+patch_size_64, p2[1]:p2[1]+patch_size_64].copy().astype(np.float32)

        p1, p2, pos32 = generate_coords_position(self.image_size, self.image_size, K=patch_size_32)
        patch1_p32 = image[p1[0]:p1[0]+patch_size_32, p1[1]:p1[1]+patch_size_32].copy().astype(np.float32)
        patch2_p32 = image[p2[0]:p2[0]+patch_size_32, p2[1]:p2[1]+patch_size_32].copy().astype(np.float32)

        p1, p2 = generate_coords_svdd(self.image_size, self.image_size, K=patch_size_64)
        patch1_s64 = image[p1[0]:p1[0]+patch_size_64, p1[1]:p1[1]+patch_size_64].copy().astype(np.float32)
        patch2_s64 = image[p2[0]:p2[0]+patch_size_64, p2[1]:p2[1]+patch_size_64].copy().astype(np.float32)
    
        p1, p2 = generate_coords_svdd(self.image_size, self.image_size, K=patch_size_32)
        patch1_s32 = image[p1[0]:p1[0]+patch_size_32, p1[1]:p1[1]+patch_size_32].copy().astype(np.float32)
        patch2_s32 = image[p2[0]:p2[0]+patch_size_32, p2[1]:p2[1]+patch_size_32].copy().astype(np.float32)

        patch1_p64 = self.ssl_transform(image=patch1_p64)['image']
        patch2_p64 = self.ssl_transform(image=patch2_p64)['image']
        patch1_p32 = self.ssl_transform(image=patch1_p32)['image']
        patch2_p32 = self.ssl_transform(image=patch2_p32)['image']
        
        patch1_s64 = self.transform(image=patch1_s64)['image']
        patch2_s64 = self.transform(image=patch2_s64)['image']
        patch1_s32 = self.transform(image=patch1_s32)['image']
        patch2_s32 = self.transform(image=patch2_s32)['image']
        
        return {
            'pos_64': (patch1_p64, patch2_p64, pos64),
            'pos_32': (patch1_p32, patch2_p32, pos32),
            'svdd_64': (patch1_s64, patch2_s64),
            'svdd_32': (patch1_s32, patch2_s32)
        }
        
    def __len__(self):
        return self.total_images * self.repeat
    
    def _build_transform(self):
        self.transform = A.Compose([
            A.Normalize(mean=MVTEC_MEAN_STD[self.category.value][0], std=MVTEC_MEAN_STD[self.category.value][1]),
            ToTensorV2()
        ])
        
        self.ssl_transform = A.Compose([
            A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.05),
            A.GaussNoise(std_range=(0.2, 0.44), p=0.05),
            A.Normalize(mean=MVTEC_MEAN_STD[self.category.value][0], std=MVTEC_MEAN_STD[self.category.value][1]),
            ToTensorV2()
        ])
    

class PatchDataset(MVTecSingle):
    def __init__(self,
                root_path: str,
                category: Union[MVTecCategory, str],
                is_train: bool,
                image_size: int = 256,
                patch_size: int = 64,
                stride: int = 16,
                cache_size: int = 32):
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        super().__init__(root_path, category=category, is_train=is_train)
        self.num_image = len(self.image_paths)
        self._load_image_and_mask = lru_cache(maxsize=cache_size)(self._load_image_and_mask)
        
    def _build_transform(self):
        self.transform_to_tensor = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=MVTEC_MEAN_STD[self.category.value][0], std=MVTEC_MEAN_STD[self.category.value][1]),
            ToTensorV2()
        ])

    def _load_image_and_mask(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
    
        image_np = np.array(Image.open(image_path).convert('RGB'))
        
        if mask_path and os.path.exists(mask_path):
            mask_np = np.array(Image.open(mask_path).convert('L'))
        else:
            h, w = image_np.shape[:2]
            mask_np = np.zeros((h, w), dtype=np.uint8)
            
        transformed = self.transform_to_tensor(image=image_np, mask=mask_np)
        full_image_tensor = transformed['image']
        full_mask_tensor = transformed['mask'].float()

        return full_image_tensor, full_mask_tensor
        
    @property
    def num_patches_y(self) -> int:
        return (self.image_size - self.patch_size) // self.stride + 1

    @property
    def num_patches_x(self) -> int:
        return (self.image_size - self.patch_size) // self.stride + 1

    def __len__(self) -> int:
        return self.num_image * self.num_patches_y * self.num_patches_x

    def __getitem__(self, idx: int):
        num_patches_per_img = self.num_patches_y * self.num_patches_x
        
        image_idx = idx // num_patches_per_img
        patch_idx_in_image = idx % num_patches_per_img
        patch_row = patch_idx_in_image // self.num_patches_x
        patch_col = patch_idx_in_image % self.num_patches_x

        full_image, full_mask = self._load_image_and_mask(image_idx)

        start_y = patch_row * self.stride
        start_x = patch_col * self.stride
        image_patch = full_image[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]

        return image_patch, image_idx, patch_row, patch_col