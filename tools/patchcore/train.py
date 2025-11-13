import argparse
from tqdm import tqdm

import torch

from model import PatchCore
from data.patchcore import MVTecPC
from utils import create_folders
from val import validate

def train(args):
    save_path = create_folders(args.project, args.category)
    print(f"Training will save to: {save_path}")
    
    model = PatchCore()
    model.to('cuda')

    train_datasets = MVTecPC(
        root_dir=args.root_path,
        category=args.category,
        is_train=True
    )
    
    test_datasets = MVTecPC(
        root_dir=args.root_path,
        category=args.category,
        is_train=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    for data in train_loader:
        images, _, _ = data
        images = images.to('cuda')
        _ = model(images)
    
    model.subsample_embedding(0.2)
    torch.save(model.state_dict(), f"{save_path}/subsampled_half.pt")
    auroc_seg, auroc_det = validate(root_path=args.root_path, category=args.category, model=model, loader=test_loader)
    print(f"Initial validation - AUROC Segmentation: {auroc_seg:.4f}, AUROC Detection: {auroc_det:.4f}")

    model.subsample_embedding(0.5)
    torch.save(model.state_dict(), f"{save_path}/subsampled_quarter.pt")
    auroc_seg, auroc_det = validate(root_path=args.root_path, category=args.category, model=model, loader=test_loader)
    print(f"Subsampled validation - AUROC Segmentation: {auroc_seg:.4f}, AUROC Detection: {auroc_det:.4f}")
    
    print("Validation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchCore Training")
    parser.add_argument("--root_path", type=str, default="/datasets/MVTec")#required=True, help="Root path of the MVTec dataset")
    parser.add_argument("--category", type=str, default="bottle")#required=True, help="MVTec category to train on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--project", type=str, default="project", help="Project name for logging")
    args = parser.parse_args()
    
    train(args)