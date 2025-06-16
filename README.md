# VLM-based Universal Deepfake Detection

A Parameter-Efficient Fine-Tuning (PEFT) approach for cross-forgery deepfake detection using Vision-Language Models (VLMs).

## 🎯 Overview

This project implements and evaluates Adapter Networks for universal deepfake detection, focusing on cross-forgery generalization. The approach uses CLIP (Contrastive Language-Image Pre-training) as the backbone with parameter-efficient fine-tuning to detect deepfakes across different generation methods.

### Key Features
- **Parameter-Efficient**: Only 589K trainable parameters (vs. millions in full fine-tuning)
- **Cross-Forgery Evaluation**: Rigorous testing on unseen forgery types
- **Dual GPU Support**: Optimized for multi-GPU training
- **Comprehensive Analysis**: Detailed performance analysis and visualization

## 📊 Results Summary

| Method | Test Accuracy | Cross-Domain Performance | Trainable Parameters |
|--------|---------------|-------------------------|---------------------|
| **Adapter Network** | **76.87%** | Balanced | 589,824 |
| Linear Probing | 70.38% | Domain Bias | 1,536 |

### Cross-Forgery Analysis
- **FaceSwap (Training Domain)**: 93.5% accuracy
- **NeuralTextures (Cross-Domain)**: 80.9% accuracy
- **Performance Gap**: +12.6% (indicates some domain bias)

## 🚀 Quick Start

### One-Command Reproduction
```bash
./run_experiment.sh
```

This script will:
1. Set up the environment
2. Verify dataset preparation
3. Train the Adapter Network (50 epochs, ~4 hours)
4. Evaluate cross-forgery performance
5. Generate analysis reports

### Manual Setup

#### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Dataset Preparation

**Download FaceForensics++ Dataset:**
```bash
# Download Real, FaceSwap, and NeuralTextures
python download_ff.py --dataset FaceForensics++ --type c23 --compression raw
```

**Prepare Cross-Forgery Split:**
```bash
# Create 80/10/10 split with paired videos
python src/prepare_cross_forgery_data.py \
    --input_dir /path/to/faceforensics++ \
    --output_dir data/ff_cross_forgery \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

Expected directory structure:
```
data/ff_cross_forgery/
├── images/
│   ├── train/
│   │   ├── n01440764/  # Real images
│   │   └── n01443537/  # Fake images (FaceSwap only)
│   ├── val/
│   └── test/           # Real + FaceSwap + NeuralTextures
├── split_info.json     # Split metadata
└── paired_videos.json # Cross-forgery pairs
```

## 🔧 Training

### Adapter Network Training
```bash
# Dual GPU training (recommended)
CUDA_VISIBLE_DEVICES=0,1 python src/train.py \
    --root data \
    --seed 17 \
    --trainer CLIP_Adapter \
    --dataset-config-file configs/datasets/ff_cross_forgery.yaml \
    --config-file configs/trainers/CoOp/vit_l14_ep50_dual_gpu.yaml \
    --output-dir results/training/clip_adapter_50epochs \
    DATASET.NUM_SHOTS 50000
```

### Training Configuration
- **Epochs**: 50 (for optimal convergence)
- **Batch Size**: 64 (32 per GPU)
- **Learning Rate**: 0.004 (with cosine scheduling)
- **Warmup**: 2 epochs
- **Optimizer**: SGD with momentum

### Expected Runtime
- **Dual GPU (TITAN RTX)**: ~4 hours
- **Single GPU**: ~7-8 hours
- **Memory Requirements**: ~12GB per GPU

## 📈 Evaluation

### Cross-Forgery Evaluation
```bash
python src/simple_paired_evaluation.py \
    --model_path results/training/clip_adapter_50epochs/clip_adapter/model.pth.tar-50 \
    --dataset_path data \
    --output_dir results/evaluation/paired_comparison
```

### Monitoring Training
```bash
# Start tensorboard
tensorboard --logdir=results/training/clip_adapter_50epochs/tensorboard --port=6006

# Open in browser
http://localhost:6006
```

## 📋 Project Structure

```
VLM_Deepfake_Detection_Project/
├── src/
│   ├── trainers/           # PEFT method implementations
│   │   ├── clip_adapter.py # Adapter Network trainer
│   │   └── ...
│   ├── models.py           # Model definitions
│   ├── train.py           # Main training script
│   ├── simple_paired_evaluation.py  # Cross-forgery evaluation
│   └── generate_final_report.py     # Report generation
├── configs/
│   ├── datasets/          # Dataset configurations
│   └── trainers/          # Training configurations
├── results/               # 
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Pre-trained weights 

```
results/training/clip_adapter/model.pth.tar-50
```

## Training log

```
results/training/log.txt
```




## 📚 References

1.S. A. Khan and D.-T. Dang-Nguyen, “CLIPping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection,” Feb. 20, 2024, arXiv: arXiv:2402.12927. doi: 10.48550/arXiv.2402.12927

