import argparse

import torch
from data.patch_svdd import MVTecSingle
from torch.utils.data import DataLoader
from tqdm import tqdm

def calculate_mean_std(root_path: str, category: str, batch_size: int = 32, num_workers: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MVTecSingle(root_path, category, is_train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(3, device=device)
    channel_sum_sq = torch.zeros(3, device=device)
    
    num_pixels = len(dataset) * 256 * 256
    
    print(f"Calculating stats for {len(dataset)} images in '{category}' category...")

    for images, _ in tqdm(loader, desc="Calculating Stats"):
        images = images.to(device)
        channel_sum += torch.sum(images, dim=[0, 2, 3])
        channel_sum_sq += torch.sum(images ** 2, dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    std = torch.sqrt((channel_sum_sq / num_pixels) - mean ** 2)

    return mean.cpu().numpy(), std.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean and standard deviation for an MVTec category.")
    parser.add_argument('--root_path', type=str, required=True, help="MVTec dataset의 루트 경로")
    parser.add_argument('--category', type=str, required=True, help="평균/표준편차를 계산할 카테고리 이름 (예: bottle, cable, carpet)")
    parser.add_argument('--batch_size', type=int, default=32, help="배치 사이즈")
    
    args = parser.parse_args()

    mean, std = calculate_mean_std(args.root_path, args.category, args.batch_size)
    
    print("\n--- Calculation Complete ---")
    print(f"Category: {args.category}")
    print(f"Mean: {mean}")
    print(f"Std Dev: {std}")
    print("--------------------------")