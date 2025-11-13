import os
from PIL import Image

import torch
from torchvision import transforms

from data.base import MVTecSingle

class MVTecEfficientAD(MVTecSingle):
    def __init__(self, root_dir: str, category: str, is_train: bool = True):
        super(MVTecEfficientAD, self).__init__(root_dir, category, is_train)
    
    def _build_transform(self):
        self.image_transfrom = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.target_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
        
        self.ae_transform = transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2)
        ])
        
    
    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.is_train:
            ae_image = self.image_transfrom(self.ae_transform(image))
            image = self.image_transfrom(image)
            return image, ae_image
        else:
            image = self.image_transfrom(image)
            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = self.target_transform(mask)
                mask = (mask > 0.5).float()
            else:
                mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
            return image, label, mask


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)