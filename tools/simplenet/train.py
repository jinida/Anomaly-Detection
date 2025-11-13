import sys
import argparse
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import SuperSimpleNet
from data.simplenet import MVTecSSN
from utils.torch_utils import to_device
from utils import create_folders
from val import validate

def train(args):
    save_path = create_folders(args.project, args.category)
    print(f"Training will save to: {save_path}")
    
    model = SuperSimpleNet()
    model = model.to('cuda')
    
    train_dataset = MVTecSSN(
        root_path=args.root_path,
        category=args.category,
        image_size=256,
        repeat=args.repeat,
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 8,
        pin_memory=True
    )
    
    best_auroc = 0.0
    
    optimizer = AdamW(
        [
            {'params': model.feature_adapter.parameters(), 'lr': 1e-4},
            {'params': model.seg_head.parameters()},
            {'params': model.cls_head.parameters()},
            {'params': model.cls_fc.parameters()},
        ],
        lr=2e-4,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90], gamma=0.4)
    test_datasets = MVTecSSN(
        root_path=args.root_path,
        category=args.category,
        image_size=256,
        repeat=1,
        is_train=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    best_auroc = 0.0
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch [{epoch + 1:03d}/{args.epochs:03d}]",
        )
        
        for data in progress_bar:
            optimizer.zero_grad()
            data = to_device(data, 'cuda')
            output = model(*data)
            loss = output
            
            loss.backward()
            optimizer.step()

            progress_bar.set_description(
                f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] Loss: {loss.item():.4f}"
            )
        scheduler.step()
        model.eval()
        auroc_seg, auroc_det = validate(
            root_path=args.root_path,
            category=args.category,
            model=model,
            loader=test_loader,
            image_size=256,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        print(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] - AUROC: {auroc_seg:.4f}, Det AUROC: {auroc_det:.4f}")
        if auroc_seg > best_auroc:
            best_auroc = auroc_seg
            print(f"New best AUROC: {best_auroc:.4f}, saving model...")
            torch.save(model.state_dict(), f"{save_path}/best.pt")
        torch.save(model.state_dict(), f"{save_path}/last.pt")
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchSVDD Training")
    parser.add_argument("--root_path", type=str, required=True, help="Root path of the MVTec dataset")
    parser.add_argument("--category", type=str, required=True, help="MVTec category to train on")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--repeat", type=int, default=3, help="Number of times to repeat the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for SVDD loss")
    parser.add_argument("--project", type=str, default="project", help="Project name for logging")
    args = parser.parse_args()
    
    train(args)