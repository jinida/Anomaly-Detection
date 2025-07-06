# PyTorch Anomaly Detection on MVTec AD
This repository is a project that uses PyTorch to implement several state-of-the-art anomaly detection models and benchmark their performance on the MVTec AD dataset.

## Implemented Models
  - **FastFlow**
  - **PatchSVDD**
  - **DifferNet**

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

## Usage
### Inference

It can be inferred using a trained model using the 'inferece.py ' script.

```bash
python tools/fast_flow/inference.py --category bottle --model_path project/bottle/001/best.pt --image_path /path/to/your/image
```

## License
This project is licensed under the Apache-2.0 License.


