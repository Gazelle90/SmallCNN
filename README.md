# SmallCNN
A CNN predicting labels for the given test images
This repo trains a CNN (scratch or pretrained torchvision model) on `./data/Task2`
and writes `submission.csv` with a `predictions` column.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yYJDnulN6nyuOqB7NYJrDACXbfp91DNA?usp=sharing)


## Quickstart (Colab)
Click the badge above. In the notebook:
1) Run the setup cell (clones this repo + installs deps).
2) Provide the data to `/content/data/Task2/` (see Data Options cell).
3) Run training; inference writes `submission.csv`.



This project was part of an ML course assignment, where the task was to train a CNN on a real dataset of 32×32 images (~5400 train / 600 test). While pretrained models like DenseNet were allowed, I wanted to build and refine my own CNN from scratch.
What started as a simple training script turned into a deep dive into optimizers, learning rates, regularization, and augmentation. Through iteration and careful tuning, I improved performance from ~50% F1 to over 70% F1.


Projects Structure 


├── train_data.npy         # Flattened training images
├── train_labels.npy       # Training labels
├── test_data.npy          # Test images
├── model.py               # CNN architecture
├── train.py               # Training & validation loop
├── utils.py               # Helpers (dataset class, transforms)
├── submission.csv         # Test predictions
└── README.md              # This file


Training Pipeline
1. Starting Simple
Implemented a basic CNN with a linear classifier.
Optimizer: Adam.
Training worked, but validation accuracy plateaued (~50%), showing overfitting.

2. Playing with Learning Rates
Tried different optimizers & LR settings:
SGD with momentum → too slow.
Adam / AdamW with various learning rates (1e-2, 1e-3, 3e-4).
Too high → divergence.
Too low → underfitting.
Sweet spot: 3e-4 with weight decay.

3. Regularization & Augmentation
To improve generalization:
RandomCrop(32, padding=4) → simulates shifted viewpoints.
RandomHorizontalFlip() → mirror invariance.
ColorJitter → robustness to lighting conditions.
RandomErasing(p=0.25) → forces model to look beyond single regions.
Normalization (CIFAR-10 mean/std) → stabilizes training.
Also used:
Weight decay.
Label smoothing.

4. Smarter Learning Rate Schedules
Instead of a fixed LR:
StepLR → reduced LR every few epochs.
ReduceLROnPlateau → adaptively lowers LR if validation loss stalls.
Later tested Cosine Annealing with warmup, giving smoother convergence.

5. Advanced Tricks for a Boost
Mixup augmentation (blends images + labels).
Exponential Moving Average (EMA) of weights.
Test-Time Augmentation (TTA) during inference.


