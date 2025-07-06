import argparse
from tqdm import tqdm

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from model import DifferNet
from data.differ_net import MVTecMultiTransform
from utils import create_folders
from tools.differ_net.val import export_gradient_maps

def loss_function(z, jacobian):
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jacobian) / z.shape[1]

def train(args):
    save_path = create_folders(args.project, args.category)
    print(f"Training will save to: {save_path}")
    
    model = DifferNet()
    optimizer = torch.optim.AdamW(model.flow.parameters(), lr=1e-4, amsgrad=True)
    model.to('cuda')
    train_datasets = MVTecMultiTransform(
        root_dir=args.root_path,
        category=args.category,
        is_train=True,
        image_size=(448, 448),
        n_transforms=4,
        fixed_rotations=False,
        repeat=args.repeat
    )
    
    test_datasets = MVTecMultiTransform(
        root_dir=args.root_path,
        category=args.category,
        is_train=False,
        image_size=(448, 448),
        n_transforms=16,
        fixed_rotations=False,
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
        model.flow.train()
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch [{epoch + 1:03d}/{args.epochs:03d}]",
        )
        
        for _, data in enumerate(progress_bar):
            optimizer.zero_grad()
            inputs, _ = data
            inputs = inputs.to('cuda')
            inputs = inputs.view(-1, *inputs.shape[-3:])
            
            z, jacobian = model(inputs)
            loss = loss_function(z, jacobian)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(
                f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] Loss: {loss.item():.4f}"
            )
            
        model.eval()
        test_loss = list()
        test_z = list()
        test_labels = list()
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f"Testing Epoch [{epoch + 1:03d}/{args.epochs:03d}]",
            )
            for _, data in enumerate(tqdm(test_loader)):
                inputs, labels = data
                inputs = inputs.to('cuda')
                inputs = inputs.view(-1, *inputs.shape[-3:])
                
                labels = labels.to('cuda')
                z, jacobian = model(inputs)
                loss = loss_function(z, jacobian)
                test_z.append(z)
                test_loss.append(loss.detach().cpu().numpy())
                test_labels.append(labels.detech().cpu().numpy())
                
                progress_bar.set_description(
                    f"Testing Epoch [{epoch + 1:03d}/{args.epochs:03d}] Loss: {loss.item():.4f}"
                )

        test_loss = np.mean(np.array(test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        z_grouped = torch.cat(test_z, dim=0).view(-1, 16, 960)
        anomaly_score = torch.mean(z_grouped ** 2, dim=(-2, -1)).cpu().numpy()
        auroc = roc_auc_score(is_anomaly, anomaly_score)
        print(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] - AUROC: {auroc:.4f}")
        
        if auroc > best_auroc:
            best_auroc = auroc
            print(f"New best AUROC: {best_auroc:.4f}, saving model...")
            torch.save(model.state_dict(), f"{save_path}/best.pt")
        torch.save(model.state_dict(), f"{save_path}/last.pt")
    
    export_gradient_maps(model, test_loader, optimizer, -1, num_transform=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchSVDD Training")
    parser.add_argument("--root_path", type=str, required=True, help="Root path of the MVTec dataset")
    parser.add_argument("--category", type=str, required=True, help="MVTec category to train on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--repeat", type=int, default=30, help="Number of times to repeat the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--project", type=str, default="project", help="Project name for logging")
    args = parser.parse_args()
    
    train(args)