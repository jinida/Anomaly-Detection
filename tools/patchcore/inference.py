from tqdm import tqdm

import cv2

import torch
from model import FastFlow
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
        model = FastFlow(8)
        model.load_state_dict(weight)
        model = model.to(device)

    model.eval().half()
    category_mean, category_std = MVTEC_MEAN_STD[category]
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype('float32') / 255.0
    image = image.transpose(2, 0, 1)
    image[0] = (image[0] - category_mean[0]) / category_std[0]
    image[1] = (image[1] - category_mean[1]) / category_std[1]
    image[2] = (image[2] - category_mean[2]) / category_std[2]
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)
    image = image.to(device).half()
    with torch.no_grad():
        output = model(image)
        anomaly_map = output['anomaly_map']
        pred_score = output['pred_score'].item()
        anomaly_map = anomaly_map.squeeze(0).squeeze(0).cpu().numpy() + 1
        anomaly_map[anomaly_map < 0.7] = 0
        anomaly_map[anomaly_map >= 0.7] = 1
        anomaly_map = (anomaly_map * 255).astype('uint8')
        anomaly_map = cv2.resize(anomaly_map, (image_size, image_size))
        
    cv2.imwrite('anomaly_map.png', anomaly_map)
    print(f"Predicted anomaly score: {1 + pred_score:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastFlow Inference")
    parser.add_argument("--image_path", type=str,default="/home/dylee/data/yjlee_temp2/datasets/bottle/train/good/002.png")#,  required=True, help="Path to the input image")
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