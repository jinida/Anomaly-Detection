import argparse

import numpy as np
import cv2
import torch

from model.patch_svdd import PatchSVDD

from data import MVTEC_MEAN_STD

def image_to_patches(
    image_tensor: torch.Tensor, 
    patch_size: int, 
    stride: int
) -> torch.Tensor:
    C = image_tensor.shape[0]

    patches = image_tensor.unfold(1, patch_size, stride)
    patches = patches.unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4)
    num_patches = patches.shape[0] * patches.shape[1]
    patches = patches.reshape(num_patches, C, patch_size, patch_size)
    
    return patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch SVDD Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input image")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category for the model")
    
    args = parser.parse_args()
    
    model = PatchSVDD()
    weight = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(weight)
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Image not found at {args.image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image.astype('float32') / 255.0
    image = image.transpose(2, 0, 1)
    image[0] = (image[0] - MVTEC_MEAN_STD[args.category][0][0]) / MVTEC_MEAN_STD[args.category][1][0]
    image[1] = (image[1] - MVTEC_MEAN_STD[args.category][0][1]) / MVTEC_MEAN_STD[args.category][1][1]
    image[2] = (image[2] - MVTEC_MEAN_STD[args.category][0][2]) / MVTEC_MEAN_STD[args.category][1][2]
    
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)
    image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    patches_64 = image_to_patches(image, patch_size=64, stride=16)
    patches_32 = image_to_patches(image, patch_size=32, stride=4)
    
    feature_64 = model(patches_64)
    feature_32 = model.layer1(patches_32)
    
    