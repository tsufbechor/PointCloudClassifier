# PointCloudClassifier

A framework for comparing state-of-the-art architectures for 3D point cloud classification. This project evaluates KPConv, PointTransformer, PointNet, and CurveCloudNet using the ModelNet10 dataset, focusing on preprocessing, training, evaluation, and visualization.

## Key Features

- **Dataset Preprocessing:**
  - Normalizes objects to a unit sphere.
  - Samples 1024 points per object.
  - Utilizes batching for efficient training.

- **Architectures Implemented:**
  - **KPConv:** Kernel Point Convolutions for local and global feature extraction.
  - **PointTransformer:** Self-attention mechanisms for effective neighborhood aggregation.
  - **PointNet:** Symmetric functions for point-wise feature extraction.
  - **CurveCloudNet:** 1D convolutions for geometric feature learning.

- **Evaluation Metrics:**
  - Accuracy, loss curves, and confusion matrices.
  - Latent space visualizations using t-SNE.
  - Class-wise precision, recall, and F1 scores.

## Results Summary

- **Best Model:** CurveCloudNet achieved the highest validation accuracy (88%) with smooth loss convergence.
- **Key Observations:**
  - CurveCloudNet and PointTransformer exhibited balanced training and validation performance.
  - KPConv showed overfitting tendencies.
  - PointNet struggled with generalization and subtle geometric differences.

## Requirements

The project dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt

## Clone the repository:
git clone https://github.com/tsufbechor/PointCloudClassifier.git
cd PointCloudClassifier

## Install dependencies:
pip install -r requirements.txt
