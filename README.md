# Unified Perception Framework
## From Pixels to Point Clouds: A Unified Deep Learning Framework for Semantic & Instance Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### 🏆 Winner - AUTOwn'25 Competition (BITS Pilani WILP)

A state-of-the-art multimodal perception system that fuses RGB camera images and LiDAR point clouds for robust semantic and instance segmentation in autonomous driving scenarios.

## ✨ Key Features

- **🔄 Multimodal Fusion**: Novel cross-attention mechanism for RGB-LiDAR integration
- **🎯 Panoptic Segmentation**: Unified semantic and instance segmentation
- **⚡ Real-time Performance**: 13.1 FPS on embedded hardware (Jetson Xavier NX)
- **🚗 Automotive Focus**: Optimized for Indian driving scenarios
- **📱 Embedded Ready**: Deployment on cost-effective hardware platforms

## 🏗️ Architecture
```
RGB Image ──► Camera Branch ──┐
                               ├──► Cross-Modal ──► Panoptic
LiDAR Points ─► LiDAR Branch ──┘     Fusion        Segmentation
```

### Core Components:
1. **Camera Branch**: DeepLabV3+ with ResNet-50 encoder
2. **LiDAR Branch**: Range-view CNN for point cloud processing  
3. **Fusion Module**: Multi-head cross-attention mechanism
4. **Panoptic Head**: Semantic + instance + offset prediction

## 📊 Performance Results

| Method | mIoU (%) | Vehicle IoU (%) | Person IoU (%) | FPS |
|--------|----------|-----------------|----------------|-----|
| Camera Only | 72.3 | 85.2 | 68.1 | 15.2 |
| **Fusion (Ours)** | **78.4** | **89.6** | **76.8** | **13.1** |
| **Improvement** | **+6.1** | **+4.4** | **+8.7** | **-2.1** |

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch
pip install open3d albumentations opencv-python
```

### Basic Usage
```python
from src.models.unified_model import UnifiedPerceptionModel

# Initialize model
model = UnifiedPerceptionModel(num_classes=19, feature_dim=256)

# Run inference
results = model(rgb_batch, lidar_list)
semantic_pred = results['semantic']
instance_pred = results['center']
```

## 📁 Repository Structure
```
unified-perception-framework/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training scripts and loops
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks and demos
├── docs/                  # Documentation
├── assets/               # Images, videos, and media
├── results/              # Experiment results and outputs
└── scripts/              # Utility scripts
```

## 🎯 Key Innovations

### 1. Cross-Modal Attention Fusion
Unlike simple concatenation, our attention mechanism allows RGB features to selectively attend to relevant LiDAR information:
```python
# Simplified fusion mechanism
Q = query_proj(camera_features)  # Camera as query
K, V = key_proj(lidar_features), value_proj(lidar_features)  # LiDAR as key-value
attention_output = scaled_dot_product_attention(Q, K, V)
```

### 2. Panoptic Segmentation Approach
Unified prediction of semantic classes, instance centers, and offset vectors:
- **Semantic Head**: Pixel-wise classification
- **Center Head**: Instance center detection  
- **Offset Head**: Pixel-to-center offset vectors

### 3. Range-View LiDAR Processing
Efficient projection of 3D point clouds to 2D range images for CNN processing:
- Preserves spatial relationships
- Enables standard 2D convolutions
- Real-time performance

## 🔧 Training

### Dataset Preparation
```bash
# Download datasets
python scripts/download_kitti.py
python scripts/download_cityscapes.py

# Preprocess data
python scripts/prepare_datasets.py
```

### Training Commands
```bash
# Camera-only baseline
python src/training/train_camera.py --config configs/camera_only.yaml

# Full unified model
python src/training/train_unified.py --config configs/unified_model.yaml
```

## 📱 Deployment

### Embedded Hardware Support
- **Jetson Xavier NX**: 13.1 FPS, 4.2GB memory
- **Jetson Nano**: 8.2 FPS, 3.1GB memory  
- **Raspberry Pi 4**: 2.1 FPS (CPU only)

### Model Optimization
```python
# TensorRT optimization
from src.utils.optimization import optimize_for_tensorrt
optimized_model = optimize_for_tensorrt(model, input_shapes)

# Quantization
from src.utils.quantization import quantize_model
quantized_model = quantize_model(model, calibration_data)
```

## 🎬 Demo & Visualization

Check out our [demo video](assets/videos/demo_comparison.mp4) showing the fusion superiority:

- **RGB-only**: Limited performance in challenging conditions
- **LiDAR-only**: Good structure but lacks semantic detail  
- **Fusion**: Best of both worlds with robust performance

## 📈 Evaluation

### Run Evaluation
```bash
python src/inference/evaluate.py --model checkpoints/unified_model.pth --dataset kitti
```

### Metrics Supported
- Mean IoU (mIoU)
- Class-wise IoU
- Panoptic Quality (PQ)
- FPS benchmarking

## 🏆 Competition Results

**AUTOwn'25 - BITS Pilani WILP (February 2026)**
- **Category**: Applied AI/ML for Automotive
- **Achievement**: 1st Place 🥇
- **Key Differentiator**: True sensor fusion vs. single-modality approaches

## 👥 Team

- **Your Name** - PhD Scholar, NIT Warangal
- **Dr. T. Ramakrishnudu** - Supervisor, NIT Warangal
- **Collaborators** - B.V. Raju Institute of Technology

## 📚 Citation

If you use this work in your research, please cite:
```bibtex
@misc{unified_perception_2026,
  title={From Pixels to Point Clouds: A Unified Deep Learning Framework for Semantic and Instance Segmentation},
  author={Your Name and Collaborators},
  year={2026},
  howpublished={AUTOwn'25 Competition, BITS Pilani WILP}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NIT Warangal for computational resources
- BITS Pilani WILP for organizing AUTOwn'25
- OpenMMLab for segmentation baselines
- KITTI and Cityscapes dataset providers

## 🔗 Links

- [Competition Details](https://bits-pilani-wilp.ac.in/autown)
- [Demo Video](assets/videos/demo.mp4)
- [Technical Paper](docs/technical_paper.pdf)
- [Poster](assets/images/competition_poster.png)

---

⭐ **Star this repository if you found it helpful!** ⭐
