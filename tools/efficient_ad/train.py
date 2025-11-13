import argparse
from tqdm import tqdm
import itertools

import torch
from data.efficient_ad import MVTecEfficientAD, InfiniteDataloader
from model import EfficientAD
from utils import create_folders
from val import validate, map_normalization

def train(args):
    save_path = create_folders(args.project, args.category)
    print(f"Training will save to: {save_path}")
    
    model = EfficientAD(out_channels=384)
    teacher_weight = torch.load(f"assets/teacher_small.pth")
    model.teacher.load_state_dict(teacher_weight)
    
    optimizer = torch.optim.Adam(itertools.chain(model.student.parameters(), model.ae.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * args.train_steps), gamma=0.1)
    
    model.to('cuda')
    
    train_datasets = MVTecEfficientAD(
        root_dir=args.root_path,
        category=args.category,
        is_train=True
    )
    train_size = int(len(train_datasets) * 0.9)
    valid_size = len(train_datasets) - train_size
    train_datasets, valid_datasets = torch.utils.data.random_split(train_datasets, [train_size, valid_size])
    test_datasets = MVTecEfficientAD(
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
    
    valid_loader = torch.utils.data.DataLoader(
        valid_datasets,
        batch_size=args.batch_size,
        shuffle=False,
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
    
    model.set_feature_params(train_loader)
    train_loader = InfiniteDataloader(train_loader)
    model.cuda()
    model.teacher.eval()
    model.student.train()
    model.ae.train()
    best_auroc = 0.0
    progress_bar = tqdm(range(args.train_steps))
    for iteration, data in zip(progress_bar, train_loader):
        optimizer.zero_grad()
        image, ae_image = data
        image = image.cuda()
        ae_image = ae_image.cuda()
        total_loss = model({'input': image, 'ae_input': ae_image})
        
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if iteration % 100 == 0:
            progress_bar.set_description(f"Iteration {iteration} - Loss: {total_loss.item():.4f}")
        
        if iteration % 1000 == 0:
            model.eval()
            maps_info = map_normalization(valid_loader, model)
            auroc_seg, auroc_det = validate(
                root_path=args.root_path,
                category=args.category,
                maps_info=maps_info,
                model=model,
                loader=test_loader,
                image_size=256,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            
            model.teacher.eval()
            model.student.train()
            model.ae.train()

            print(f"Iteration [{iteration} / {args.train_steps}] - AUROC(Seg): {auroc_seg:.4f}, AUROC(Det): {auroc_det:.4f}")
            
            model.q_st_start = maps_info[0]
            model.q_st_end = maps_info[1]
            model.q_ae_start = maps_info[2]
            model.q_ae_end = maps_info[3]
            
            if auroc_seg > best_auroc:
                best_auroc = auroc_seg
                print(f"New best AUROC: {best_auroc:.4f}, saving model...")
                torch.save(model.state_dict(), f"{save_path}/best.pt")
            torch.save(model.state_dict(), f"{save_path}/last.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientAD Training")
    parser.add_argument("--root_path", type=str, required=True, help="Root path of the MVTec dataset")
    parser.add_argument("--category", type=str, required=True, help="MVTec category to train on")
    parser.add_argument("--train_steps", type=int, default=75000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--project", type=str, default="project", help="Project name for logging")
    args = parser.parse_args()
    
    train(args)