import argparse
from tqdm import tqdm

import torch

from model import FastFlow
from data.fast_flow import MVTecFF
from utils import create_folders
from val import validate

def train(args):
    save_path = create_folders(args.project, args.category)
    print(f"Training will save to: {save_path}")
    
    model = FastFlow(8, 256, 1.0)
    optimizer = torch.optim.AdamW(model.flows.parameters(), lr=1e-4, amsgrad=True)
    model.to('cuda')
    
    train_datasets = MVTecFF(
        root_dir=args.root_path,
        category=args.category,
        is_train=True,
        image_size=(256, 256),
        repeat=args.repeat
    )
    
    test_datasets = MVTecFF(
        root_dir=args.root_path,
        category=args.category,
        is_train=False,
        image_size=(256, 256),
        repeat=1
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
    
    best_auroc = 0.0
    for epoch in range(args.epochs):
        model.flows.train()
        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch [{epoch + 1:03d}/{args.epochs:03d}]",
        )
        
        for data in progress_bar:
            optimizer.zero_grad()
            inputs = data.to('cuda')
            output = model(inputs)
            loss = output['loss']
            
            loss.backward()
            optimizer.step()

            progress_bar.set_description(
                f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] Loss: {loss.item():.4f}"
            )
            
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
        
        print(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] - AUROC: {auroc_seg:.4f}")
        if auroc_seg > best_auroc:
            best_auroc = auroc_seg
            print(f"New best AUROC: {best_auroc:.4f}, saving model...")
            torch.save(model.state_dict(), f"{save_path}/best.pt")
        torch.save(model.state_dict(), f"{save_path}/last.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastFlow Training")
    parser.add_argument("--root_path", type=str, required=True, help="Root path of the MVTec dataset")
    parser.add_argument("--category", type=str, required=True, help="MVTec category to train on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--repeat", type=int, default=100, help="Number of times to repeat the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--project", type=str, default="project", help="Project name for logging")
    args = parser.parse_args()
    
    train(args)