from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from ignite.contrib import metrics
from data.simplenet import MVTecSSN
from data.utils import MVTecCategory
from model import SuperSimpleNet

def validate(
    root_path: str,
    category: MVTecCategory,
    model: SuperSimpleNet = None,
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
        model = SuperSimpleNet(**weight['args'])
        model.load_state_dict(weight['state_dict'])
        model = model.to(device)
    
    if loader is None:
        datasets = MVTecSSN(root_dir=root_path, category=category, is_train=False, image_size=(image_size, image_size), repeat=1)
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
    
    with torch.no_grad():
        progress_bar = tqdm(
            loader,
            total=len(loader),
            desc=f"[val]"
        )
        
        for image, target, is_anomal in progress_bar:
            image = image.to(device)
            
            output = model(image)
            anomaly_map, anomaly_score = output
            target = target.to(device).long()
            metric_seg.update((anomaly_map.flatten(), target.flatten()))
            metric_det.update((anomaly_score, is_anomal.long()))
            
        auroc_seg = metric_seg.compute()
        auroc_det = metric_det.compute()
    
    return auroc_seg, auroc_det