import numpy as np
import cv2

import torch
from model import SuperSimpleNet
from data import MVTEC_MEAN_STD

def inference(
    image_path: str,
    model_path: str,
    category: str = 'bottle',
    image_size: int = 256
):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_path:
        weight = torch.load(model_path)
        model = SuperSimpleNet()
        model.load_state_dict(weight)
        model = model.to(device)

    model.eval()
    category_mean, category_std = MVTEC_MEAN_STD[category]
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    ori_image = cv2.resize(image.copy(), (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    
    
    image = image.astype('float32') / 255.0
    image = image.transpose(2, 0, 1)
    image[0] = (image[0] - category_mean[0]) / category_std[0]
    image[1] = (image[1] - category_mean[1]) / category_std[1]
    image[2] = (image[2] - category_mean[2]) / category_std[2]
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    
    with torch.no_grad():
        output = model(image)
        anomaly_map_tensor, anomal_score = output

        anomaly_map_processed = anomaly_map_tensor.squeeze().sigmoid()
        
        anomaly_map_processed[anomaly_map_processed >= 0.95] = 1
        anomaly_map_processed[anomaly_map_processed < 0.95] = 0
        
        anomaly_map_numpy = anomaly_map_processed.cpu().numpy()

        anomaly_map_final = (anomaly_map_numpy * 255).astype(np.uint8)
        anomal_score_item = anomal_score.sigmoid().item()
        
    overlay_color = np.array([128, 0, 0], dtype=np.uint8)
    anomaly_mask = (anomaly_map_final == 255)
    mask_indices = np.where(anomaly_mask)
    region_to_overlay = ori_image[mask_indices[0], mask_indices[1]]
    alpha = 0.5
    beta = 0.5 
    mixed_color = (region_to_overlay.astype(np.float32) * alpha + overlay_color.astype(np.float32) * beta).astype(np.uint8)
    ori_image[mask_indices[0], mask_indices[1]] = mixed_color
    
    cv2.imwrite('anomaly_map.png', ori_image)
    print(f"Predicted anomaly score: {anomal_score_item:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastFlow Inference")
    parser.add_argument("--image_path", type=str,default="datasets/MVTec/bottle/test/contamination/000.png")#,  required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category for the model")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input image")
    
    args = parser.parse_args()
    
    inference(
        image_path=args.image_path,
        model_path=args.model_path,
        category=args.category,
        image_size=args.image_size
    )
    print("Inference complete. Anomaly map saved as 'anomaly_map.png'.")