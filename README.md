# Solar Panel Defect Detection with YOLO26n + PATConv + LWGA

This repository provides the implementation of the solar panel defect detection method described in the paper.

---

## Overview

- **Base model**: YOLO26n
- **Improvements**:
  - **PATConv** (Partial Attention Convolution) replaces all standard 3x3 convolutions.
  - **LWGA** (Lightweight Grouped Attention) is inserted after every C3k2 module.
- **Strategy**: spatial focusing first, then channel enhancement.
- **Dataset**: 6,079 visible-light images of solar panels with three classes: broken, dirty, clean.
- **Test set results**: mAP@0.5 = 97.3%, mAP@0.5:0.95 = 97.3%.

---

## Environment

- Python 3.11
- PyTorch 2.4.1
- Ultralytics 8.4.10
- See requirements.txt for the full list of dependencies.

---

## Data Preparation

1. Download the dataset and organize it as follows:

```

dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
├── train/
├── val/
└── test/

```

2. Copy configs/data.yaml.example to configs/data.yaml and modify the path field to point to your dataset directory.

---

## Training

Run the following command to train the model with the same configuration as in the paper:

```

python train.py --data configs/data.yaml --epochs 120 --batch 8 --imgsz 640 --device 0

```

**Note**: If you encounter GPU memory limitations, you may reduce the batch size (e.g., --batch 4). However, the default setting in the code is batch=8, which matches the paper.

---

## Test Evaluation

After training, evaluate the best model on the test set:

```

python eval_test.py

```

The script will load best.pt, run evaluation on the test set (609 images), and print the metrics.

---

## Overall Architecture

The framework follows the spatial focusing first, then channel enhancement strategy. PATConv replaces all standard 3x3 convolutions, and LWGA is inserted after each C3k2 module.

---

## Results

All reported metrics are obtained on the test set using the best checkpoint. The model achieves:

- mAP@0.5: 97.3%
- mAP@0.5:0.95: 97.3%

For detailed per-class results and ablation studies, please refer to the paper.

---

## Citation

If you use this code in your research, please cite the original paper.

---

## License

This project is licensed under the AGPL-3.0 License, due to the dependency on Ultralytics. See the LICENSE file for details.