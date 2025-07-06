import os
from typing import List, Union

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import MVTEC_MEAN_STD
from data.utils import MVTecCategory

class MVTecBase(Dataset):
    def __init__(self, root_dir, categories: List[MVTecCategory], is_train: bool = False):
        self.root_dir = root_dir
        self.categories = [cat if isinstance(cat, MVTecCategory) else MVTecCategory(cat.lower()) for cat in categories]
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        self.image_transfrom = None
        self.target_transform = None
        self.total_images = 0
        
        self._load_dataset()
        self._build_transform()
    
    def _build_transform(self):
        self.image_transfrom = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=MVTEC_MEAN_STD[self.categories[0].value][0], std=MVTEC_MEAN_STD[self.categories[0].value][1]),
            ToTensorV2()
        ])
        
        self.target_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            ToTensorV2()
        ])
    
    def _load_dataset(self):
        for category in self.categories:
            category_path = os.path.join(self.root_dir, category.value)

            if not os.path.isdir(category_path):
                print(f"Warning: Category path not found: {category_path}. Skipping.")
                continue
            
            if self.is_train:
                normal_dir = os.path.join(category_path, 'train', 'good')
                if os.path.isdir(normal_dir):
                    img_names = [f for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    self.image_paths.extend([os.path.join(normal_dir, name) for name in img_names])
                    self.labels.extend([0] * len(img_names))
                    self.mask_paths.extend([None] * len(img_names))
            else:
                test_dir = os.path.join(category_path, 'test')
                if os.path.isdir(test_dir):
                    for defect_type in os.listdir(test_dir):
                        defect_path = os.path.join(test_dir, defect_type)
                        if os.path.isdir(defect_path):
                            label = 0 if defect_type == 'good' else 1
                            img_names = [f for f in os.listdir(defect_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                            
                            self.image_paths.extend([os.path.join(defect_path, name) for name in img_names])
                            self.labels.extend([label] * len(img_names))
                            
                            if label == 0:
                                self.mask_paths.extend([None] * len(img_names))
                            else:
                                mask_dir = os.path.join(category_path, 'ground_truth', defect_type)
                                self.mask_paths.extend([os.path.join(mask_dir, name.rsplit('.', 1)[0] + '_mask.png') for name in img_names])
        
        self.total_images = len(self.image_paths)
        
    def __len__(self) -> int:
        return self.total_images
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        image = np.array(Image.open(image_path).convert('RGB'))

        if self.image_transfrom:
            image = self.image_transfrom(image=image)['image']
        
        if mask_path and os.path.exists(mask_path):
            mask_np = np.array(Image.open(mask_path).convert('L'))
            if self.target_transform:
                transformed_mask = self.target_transform(image=mask_np)
                mask = transformed_mask['image']
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros_like(image, dtype=torch.float32)
        
        return image, label, mask

class MVTecAll(MVTecBase):
    def __init__(self, root_dir, is_train: bool = False):
        categories = list(MVTecCategory)
        super().__init__(root_dir, categories, is_train)

class MVTecSingle(MVTecBase):
    def __init__(self, root_dir, category: Union[MVTecCategory, str], is_train: bool = False):
        if isinstance(category, str):
            category = MVTecCategory(category.lower())
        elif not isinstance(category, MVTecCategory):
            raise ValueError("category must be a MVTecCategory enum or a string.")
        
        self.category = category
        super().__init__(root_dir, [category], is_train)

class MVTecMultiple(MVTecBase):
    def __init__(self, root_dir, categories: Union[List[MVTecCategory], List[str]], is_train: bool = False):
        super().__init__(root_dir, categories, is_train)