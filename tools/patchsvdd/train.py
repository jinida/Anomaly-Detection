import sys
sys.path.append("..")
import argparse
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import PatchSVDD
from data.patch_svdd import MVTecCategory, PatchSSLDataset
from utils.torch_utils import to_device
from utils import create_folders
from val import validate

def train(args):
    save_folder = create_folders(args.project, args.category)
    print(f"Training will save to: {save_folder}")
    
    model = PatchSVDD()
    model = model.to('cuda')
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    train_dataset = PatchSSLDataset(
        root_path=args.root_path,
        category=MVTecCategory(args.category),
        image_size=256,
        repeat=args.repeat
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 8,
        pin_memory=True
    )
    
    best_auroc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        
        progress_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch [{epoch + 1:03d}/{args.epochs:03d}]",
        )
        
        for _, batch in progress_bar:
            batch = to_device(batch, 'cuda')
            optimizer.zero_grad()
            
            outputs = model(batch)
            
            pos64, pos32 = outputs['pos_64'], outputs['pos_32']
            svdd64, svdd32 = outputs['svdd_64'], outputs['svdd_32']
            
            position_loss = args.alpha * (pos64 + pos32)
            svdd_loss = svdd64 + svdd32
            loss = position_loss + svdd_loss
            
            loss.backward()
            optimizer.step()
            progress_bar.set_description(
                f"Epoch [{epoch + 1:03d}/{args.epochs}] - "
                f"Position Loss: {position_loss.item():.5f}, "
                f"SVDD Loss: {svdd_loss.item():.5f}"
            )
        
        print(f"Epoch [{epoch + 1:03d}/{args.epochs}] - Validating model...")
        auroc = validate(root_path=args.root_path,
                        category=MVTecCategory(args.category),
                        model=model,
                        image_size=256,
                        k=1,
                        batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
                        num_workers=args.num_workers if hasattr(args, 'num_workers') else 8)
        
        print(f"Epoch [{epoch + 1:03d}/{args.epochs}] - AUROC Results:")
        print(f"Detection AUROC: {auroc['Detection AUROC (Sum)']:.4f}, Segmentation AUROC: {auroc['Segmentation AUROC (Sum)']:.4f}")
        print(f"Detection AUROC (64): {auroc['Detection AUROC (64)']:.4f}, Segmentation AUROC (64): {auroc['Segmentation AUROC (64)']:.4f}")
        print(f"Detection AUROC (32): {auroc['Detection AUROC (32)']:.4f}, Segmentation AUROC (32): {auroc['Segmentation AUROC (32)']:.4f}")
        
        if auroc['Detection AUROC (Sum)'] > best_auroc:
            best_auroc = auroc['Detection AUROC (Sum)']
            torch.save(model.state_dict(), f"{save_folder}/best_{args.category}.pt")
            print(f"Best model saved with AUROC: {best_auroc:.4f}")
        
        torch.save(model.state_dict(), f"{save_folder}/last_{args.category}.pt")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchSVDD Training")
    parser.add_argument("--root_path", type=str, required=True, help="Root path of the MVTec dataset")
    parser.add_argument("--category", type=str, required=True, help="MVTec category to train on")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--repeat", type=int, default=100, help="Number of times to repeat the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for SVDD loss")
    parser.add_argument("--project", type=str, default="project", help="Project name for logging")
    args = parser.parse_args()
    
    train(args)