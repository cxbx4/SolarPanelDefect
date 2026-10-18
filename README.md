# Solar Panel Defect Detection with YOLO26n + PATConv + LWGA Implementation and Training

## Project Overview
This project implements an improved object detection model based on YOLO26n for solar panel surface defect detection tasks. It introduces the PATConv and LWGA modules to enhance small-defect detection and multi-scale feature extraction. The repository includes complete functionalities such as model definition, data processing, training workflow, and performance evaluation, supporting training and testing on a custom solar panel dataset.

## Environment Dependencies
- Python 3.11
- PyTorch == 2.4.1
- Ultralytics == 8.4.10

## Model Architecture
The model is improved based on the YOLO26n architecture, with the following main features:

1. Replaces all standard 3×3 convolutions with **PATConv** (Partial Attention Convolution) in the backbone, neck, and detection head to reduce background computation and enhance small-defect features.
2. Inserts **LWGA** (Lightweight Grouped Attention) after every C3k2 module to achieve efficient multi-scale channel enhancement.
3. Follows the "spatial focusing first, then channel enhancement" strategy.
4. Maintains lightweight characteristics (2.79M parameters, 6.8 GFLOPs), suitable for edge device deployment.

The model definition and related files are located in the project root, which mainly include:
- `custom_modules.py`: Definition of `PATConv` and `LWGA` modules.
- `register_modules.py`: Registration of LWGA into the Ultralytics framework.
- `yolo26n_patconv_lwga.yaml`: Model structure configuration.
- `train.py`: Training script with recursive PATConv replacement logic.
- `eval_test.py`: Test set evaluation script.

## Dataset Preparation
1. The dataset should be organized according to the following directory structure:

```plaintext
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

2. train, val, and test correspond to the training set (4,254 images), validation set (1,216 images), and test set (609 images) respectively, with a 7:2:1 split.
3. The dataset contains 6,079 visible-light images covering three classes: broken, dirty, and clean.

Data Preprocessing

In this project, the data preprocessing pipeline is handled automatically by the Ultralytics framework. The strategy is configured in the training script and data.yaml. Different strategies are adopted for the training set and validation/test set.

Preprocessing Pipeline

1. Training Set Preprocessing

Data augmentation techniques are applied to the training set to improve the generalization ability of the model. In train.py, the following parameters are set:

```python
mosaic=1.0,
fliplr=0.5,
hsv_h=0.015,
hsv_s=0.7,
hsv_v=0.4,
```

2. Validation/Test Set Preprocessing

A simpler preprocessing is used for the validation and test sets without data augmentation. The input images are uniformly resized to 640×640 pixels.

Explanation of Preprocessing Details

1. Size Adjustment: All images are resized to 640×640 to ensure consistent input dimensions.
2. Data Augmentation: Only applied to the training set (Mosaic, flip, HSV) to prevent overfitting.
3. Normalization: Handled internally by the Ultralytics framework.
4. Data Loading: Managed by the Ultralytics build_dataloader function, reading from the standard YOLO format folder structure.

Training the Model

Before training, copy configs/data.yaml.example to configs/data.yaml and modify the path field to point to your dataset directory.

Run the training script:

```bash
python train.py --data configs/data.yaml --epochs 120 --batch 8 --imgsz 640 --device 0
```

The main parameters during training can be adjusted in the argparse section of train.py:

· --batch: Batch size, default 8 (matching the paper).
· --epochs: Number of training epochs, default 120.
· --imgsz: Input image size, default 640.

During training, the following will be done automatically:

· Replace all standard 3×3 convolutions with PATConv and insert LWGA after each C3k2.
· Save the best model (based on validation set performance).
· Record training/validation loss and mAP metrics.
· Use the MuSGD optimizer with an initial learning rate of 0.01 and cosine annealing.

Result Output

Training results will be saved in the runs/detect/runs/train/yolo26n_patconv_lwga/ folder:

· weights/best.pt: The best model weights.
· weights/last.pt: The last epoch model weights.
· results.csv: Records of loss and mAP during training.
· confusion_matrix.png: Confusion matrix visualization.

To evaluate the model on the test set, run:

```bash
python eval_test.py
```

The evaluation metrics will be printed to the console.

Model Evaluation Metrics

Model evaluation includes the following metrics:

· Precision (P)
· Recall (R)
· mAP@0.5
· mAP@0.5:0.95

The proposed model achieves the following results on the test set (609 images):

· mAP@0.5: 97.3%
· mAP@0.5:0.95: 97.3%

Customization and Extension

1. To modify the model structure, adjust the network definition in yolo26n_patconv_lwga.yaml.
2. To change the data augmentation strategy, modify the mosaic, fliplr, hsv_h, hsv_s, and hsv_v parameters in train.py.
3. To adjust training parameters (learning rate, optimizer, etc.), modify the relevant settings in the model.train() method.
Note
1. Ensure that the ultralytics, torch, and torchvision packages are installed before running.
2. The runs folder and subdirectories will be created automatically during training.
3. For Windows users: When running eval_test.py, ensure workers=0 is set to avoid multiprocessing errors. The script must be wrapped in if __name__ == '__main__':.