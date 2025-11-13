from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from ignite.contrib import metrics

from data.base import MVTecCategory
from data.efficient_ad import MVTecEfficientAD
from model.efficient_ad import EfficientAD

def map_normalization(loader, model):
    maps_st = []
    maps_ae = []
    with torch.no_grad():
        for image, _ in loader:
            map_st, map_ae = model(image.cuda())
            maps_st.append(map_st)
            maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

def validate(
    root_path: str,
    category: MVTecCategory,
    maps_info: tuple,
    model: EfficientAD = None,
    model_path: str = None,
    loader: DataLoader = None,
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = 'cuda'
):
    
    if model is None and model_path is None:
        raise ValueError("Either model or model_path must be provided.")
    
    if model_path:
        weight = torch.load(model_path)
        model = EfficientAD(384)
        model.load_state_dict(weight['state_dict'])
        model = model.to(device)
    
    if loader is None:
        datasets = MVTecEfficientAD(root_dir=root_path, category=category, is_train=False)
        loader = DataLoader(
            datasets,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    model.eval()
    metric_seg = metrics.ROC_AUC()
    metric_det = metrics.ROC_AUC()
    q_st_start, q_st_end, q_ae_start, q_ae_end = maps_info
    
    with torch.no_grad():
        for image, target, is_anomal in loader:
            image = image.to(device)
            
            map_st, map_ae = model(image)
            target = target.to(device).long()
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
            anomaly_map = 0.5 * map_st + 0.5 * map_ae
            anomaly_map = torch.nn.functional.pad(anomaly_map, (4, 4, 4, 4))
            anomaly_map = torch.nn.functional.interpolate(anomaly_map, size=image.shape[2:], mode='bilinear', align_corners=False)
            pred_score = torch.max(anomaly_map)
            
            metric_seg.update((anomaly_map.flatten(), target.flatten()))
            metric_det.update((pred_score.flatten(), is_anomal.long().flatten()))
            
        auroc_det = metric_det.compute()
        auroc_seg = metric_seg.compute()
        
        print(f"Validation AUROC(Seg): {auroc_seg:.4f}")
        print(f"Validation AUROC(Det): {auroc_det:.4f}")
    
    return auroc_seg, auroc_det