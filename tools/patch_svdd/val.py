from tqdm import tqdm
import argparse

import numpy as np
import cv2
import numba

from sklearn.metrics import roc_auc_score
import hnswlib

import torch
from torch.utils.data import DataLoader

from model.patch_svdd import PatchSVDD
from data.patch_svdd import PatchDataset, MVTecCategory

def detection_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2: return 0.5
    return roc_auc_score(y_true, y_score)

def segmentation_auroc(gt_masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    gt_masks_binary = (gt_masks > 0).astype(np.uint8)
    if len(np.unique(gt_masks_binary)) < 2: return 0.5
    if anomaly_maps.shape[-2:] != gt_masks_binary.shape[-2:]:
        target_shape = gt_masks_binary.shape[-2:]
        resized_maps = np.array([cv2.resize(m, (target_shape[1], target_shape[0]), cv2.INTER_LINEAR) for m in anomaly_maps])
    else:
        resized_maps = anomaly_maps
    return roc_auc_score(gt_masks_binary.flatten(), resized_maps.flatten())

def measure_emb_NN(emb_te, emb_tr, k=1):
    D = emb_tr.shape[-1]
    train_emb_flat = emb_tr.reshape(-1, D).astype('float32')
    test_emb_flat = emb_te.reshape(-1, D).astype('float32')
    test_shape = emb_te.shape
    num_train_elements = len(train_emb_flat)
    
    index = hnswlib.Index(space='l2', dim=D)

    index.init_index(max_elements=num_train_elements, ef_construction=200, M=16)
    index.add_items(train_emb_flat)
    
    index.set_ef(32)
    
    _, distances = index.knn_query(test_emb_flat, k=k)
    
    return np.mean(distances, axis=-1).reshape(test_shape[:-1])

@numba.njit(parallel=True, fastmath=True)
def distribute_scores(score_masks, output_shape, K, S):
    N, H, W = score_masks.shape[0], output_shape[0], output_shape[1]
    mask = np.zeros((N, H, W), dtype=np.float32)
    cnt = np.zeros((N, H, W), dtype=np.int32)
    I, J = score_masks.shape[1:3]

    for n in numba.prange(N):
        for i in range(I):
            for j in range(J):
                h, w = i * S, j * S
                patch_area = mask[n, h:h+K, w:w+K]
                patch_area += score_masks[n, i, j]
                
                cnt_area = cnt[n, h:h+K, w:w+K]
                cnt_area += 1

    for n in numba.prange(N):
        for i in range(H):
            for j in range(W):
                if cnt[n, i, j] == 0:
                    cnt[n, i, j] = 1
    
    return mask / cnt

def validate(root_path: str,
            category: MVTecCategory,
            model: PatchSVDD=None,
            model_path: str = None,
            image_size: int = 256,
            k=1,
            batch_size: int = 32,
            num_workers: int = 8,
            device: str = 'cuda') -> float:
    
    if model is None and model_path is None:
        raise ValueError("Either model or model_path must be provided.")
    
    if model_path:
        weight = torch.load(model_path)
        model = PatchSVDD()
        model.load_state_dict(weight)
        model = model.to(device)
        
    model.eval()

    loaders = {}
    datasets = {}
    for is_train in [True, False]:
        for patch_size in [64, 32]:
            stride = 16 if patch_size == 64 else 4
            key = f"{'train' if is_train else 'test'}_{patch_size}"

            dataset = PatchDataset(
                root_path=root_path, category=category, is_train=is_train,
                image_size=image_size, patch_size=patch_size, stride=stride,
            )
            datasets[key] = dataset
            loaders[key] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = datasets['test_64']
    y_true_labels = np.array(test_dataset.labels, dtype=np.int32)
    
    gt_masks = np.array([test_dataset._load_image_and_mask(i)[1].numpy() for i in range(test_dataset.num_image)])
    
    embeddings = {}
    with torch.no_grad():
        for key, loader in loaders.items():
            dataset = loader.dataset
            patch_size = dataset.patch_size
            embed = torch.zeros((len(dataset.image_paths), dataset.num_patches_y, dataset.num_patches_x, model.dimension))
            progress_bar = tqdm(loader, desc=f"Processing {key} patches", total=len(loader))
            for patches, image_indices, patch_rows, patch_cols in progress_bar:
                patches = patches.to(device)
                if patch_size == 32:
                    features = model.layer1(patches)
                else:
                    features = model(patches)
                
                for feature, n, i, j in zip(features.cpu(), image_indices, patch_rows, patch_cols):
                    embed[n, i, j] = torch.squeeze(feature)
            
            embeddings[key] = embed.numpy()
            
    embs64 = (embeddings['train_64'], embeddings['test_64'])
    embs32 = (embeddings['train_32'], embeddings['test_32'])

    print("Calculating anomaly maps...")
    maps_64 = measure_emb_NN(embs64[1], embs64[0], k=k)
    maps_64 = distribute_scores(maps_64, (image_size, image_size), K=64, S=16)
    scores_64 = maps_64.max(axis=-1).max(axis=-1)
    det_64 = detection_auroc(y_true_labels, scores_64)
    seg_64 = segmentation_auroc(gt_masks, maps_64)

    maps_32 = measure_emb_NN(embs32[1], embs32[0], k=k)
    maps_32 = distribute_scores(maps_32, (image_size, image_size), K=32, S=4)
    scores_32 = maps_32.max(axis=-1).max(axis=-1)
    det_32 = detection_auroc(y_true_labels, scores_32)
    seg_32 = segmentation_auroc(gt_masks, maps_32)

    maps_sum = maps_64 + maps_32
    scores_sum = maps_sum.max(axis=-1).max(axis=-1)
    det_sum = detection_auroc(y_true_labels, scores_sum)
    seg_sum = segmentation_auroc(gt_masks, maps_sum)
    
    return {
        'Detection AUROC (64)': det_64, 'Segmentation AUROC (64)': seg_64,
        'Detection AUROC (32)': det_32, 'Segmentation AUROC (32)': seg_32,
        'Detection AUROC (Sum)': det_sum, 'Segmentation AUROC (Sum)': seg_sum,
    }
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate PatchSVDD model on MVTec dataset.")
    parser.add_argument('--root_path', type=str, default="datasets", help="Root path of the MVTec dataset")
    parser.add_argument('--category', type=str, default="bottle", help="Category name (e.g., 'bottle', 'cable')")
    parser.add_argument('--model_path', type=str, default="project/bottle/065/best_bottle.pt", help="Path to the trained model weights")
    parser.add_argument('--image_size', type=int, default=256, help="Image size for validation")
    parser.add_argument('--k', type=int, default=1, help="Number of nearest neighbors for anomaly detection")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")

    args = parser.parse_args()

    category = MVTecCategory(args.category)
    
    results = validate(
        root_path=args.root_path,
        category=category,
        model_path=args.model_path,
        image_size=args.image_size,
        k=args.k,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("\n--- Validation Results ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    print("--------------------------")