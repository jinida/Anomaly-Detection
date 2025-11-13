# PyTorch Anomaly Detection on MVTec AD
This repository is a project that uses PyTorch to implement several state-of-the-art anomaly detection models and benchmark their performance on the MVTec AD dataset.

## Implemented Models
  - **FastFlow** - Normalizing flow-based unsupervised anomaly detection
  - **PatchSVDD** - Patch-based Support Vector Data Description
  - **DifferNet** - Difference-based anomaly detection with normalizing flows
  - **EfficientAD** - Efficient anomaly detection model
  - **PatchCore** - Memory-bank based anomaly detection
  - **SimpleNet** - Lightweight anomaly detection network

## Dataset Preparation

1. [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. Place the dataset in the folder structure as below.
    ```
    /path/to/your/dataset/
    ├── bottle/
    │   ├── train/
    │   │   └── good/
    │   │       ├── 000.png
    │   │       └── ...
    │   └── test/
    │       ├── good/
    │       │   └── 000.png
    │       ├── broken_large/
    │       │   └── 000.png
    │       └── ...
    ├── cable/
    │   └── ...
    └── ...
    ```

## Project Structure
```
anomaly/
├── model/              # Model implementations
│   ├── fast_flow.py
│   ├── patch_svdd.py
│   ├── differ_net.py
│   ├── efficient_ad.py
│   ├── patchcore.py
│   ├── simplenet.py
│   └── filter.py
├── data/              # Data loaders for each model
│   ├── base.py
│   ├── fast_flow.py
│   ├── patch_svdd.py
│   ├── differ_net.py
│   ├── efficient_ad.py
│   ├── patchcore.py
│   └── simplenet.py
├── tools/             # Training, validation, and inference scripts
│   ├── calc_mean_std.py
│   ├── fast_flow/
│   │   ├── train.py
│   │   ├── val.py
│   │   └── inference.py
│   ├── patchsvdd/
│   │   ├── train.py
│   │   ├── val.py
│   │   └── inference.py
│   ├── differ_net/
│   │   ├── train.py
│   │   └── val.py
│   ├── efficient_ad/
│   │   ├── train.py
│   │   └── val.py
│   ├── patchcore/
│   │   ├── train.py
│   │   ├── val.py
│   │   └── inference.py
│   └── simplenet/
│       ├── train.py
│       ├── val.py
│       └── inference.py
└── utils/             # Utility functions
    ├── torch_utils.py
    └── generate.py
```

## Usage

### Training

Each model has its own training script in the `tools/` directory:

```bash
# FastFlow
python tools/fast_flow/train.py --category bottle --data_path /path/to/mvtec

# PatchSVDD
python tools/patchsvdd/train.py --category bottle --data_path /path/to/mvtec

# DifferNet
python tools/differ_net/train.py --category bottle --data_path /path/to/mvtec

# EfficientAD
python tools/efficient_ad/train.py --category bottle --data_path /path/to/mvtec

# PatchCore
python tools/patchcore/train.py --category bottle --data_path /path/to/mvtec

# SimpleNet
python tools/simplenet/train.py --category bottle --data_path /path/to/mvtec
```

### Validation

Validate trained models using validation scripts:

```bash
# FastFlow
python tools/fast_flow/val.py --category bottle --data_path /path/to/mvtec --model_path /path/to/checkpoint

# PatchSVDD
python tools/patchsvdd/val.py --category bottle --data_path /path/to/mvtec --model_path /path/to/checkpoint

# DifferNet
python tools/differ_net/val.py --category bottle --data_path /path/to/mvtec --model_path /path/to/checkpoint

# EfficientAD
python tools/efficient_ad/val.py --category bottle --data_path /path/to/mvtec --model_path /path/to/checkpoint

# PatchCore
python tools/patchcore/val.py --category bottle --data_path /path/to/mvtec --model_path /path/to/checkpoint

# SimpleNet
python tools/simplenet/val.py --category bottle --data_path /path/to/mvtec --model_path /path/to/checkpoint
```

### Inference

Run inference on single images using trained models:

```bash
# FastFlow
python tools/fast_flow/inference.py --category bottle --model_path project/bottle/001/best.pt --image_path /path/to/your/image

# PatchSVDD
python tools/patchsvdd/inference.py --category bottle --model_path project/bottle/001/best.pt --image_path /path/to/your/image

# PatchCore
python tools/patchcore/inference.py --category bottle --model_path project/bottle/001/best.pt --image_path /path/to/your/image

# SimpleNet
python tools/simplenet/inference.py --category bottle --model_path project/bottle/001/best.pt --image_path /path/to/your/image
```

## Utilities

### Calculate Dataset Mean and Standard Deviation

```bash
python tools/calc_mean_std.py --data_path /path/to/mvtec --category bottle
```

## License
This project is licensed under the Apache-2.0 License.


