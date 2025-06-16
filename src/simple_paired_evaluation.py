#!/usr/bin/env python3
"""
Simple Paired Comparison Evaluation
Uses the same model structure as training for evaluation
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
import glob
from tqdm import tqdm
from collections import defaultdict
import clip


class CLIPLinearModel(nn.Module):
    """CLIP + Linear classifier model (same as training)"""
    
    def __init__(self, clip_model_name="ViT-L/14", num_classes=2):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name, device='cpu')
        self.clip_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Linear classifier
        self.classifier = nn.Linear(768, num_classes)  # ViT-L/14 has 768 features
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            features = features.float()
        
        features = self.dropout(features)
        return self.classifier(features)


def load_split_info(data_dir):
    """Load split information including paired videos"""
    split_file = os.path.join(data_dir, "split_info.json")
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            return json.load(f)
    return None


def extract_video_id_from_filename(filename):
    """Extract video ID from filename"""
    parts = filename.split('_')
    if len(parts) >= 3:
        if parts[0] in ['faceswap', 'neuraltextures']:
            return f"{parts[1]}_{parts[2]}"
    return None


def group_test_images_by_video(test_dir):
    """Group test images by video ID and method"""
    grouped_images = defaultdict(lambda: {'faceswap': [], 'neuraltextures': [], 'real': []})

    # Get fake test images
    fake_test_dir = os.path.join(test_dir, "n01443537")
    if os.path.exists(fake_test_dir):
        image_files = glob.glob(os.path.join(fake_test_dir, "*.png")) + \
                      glob.glob(os.path.join(fake_test_dir, "*.jpg"))

        for img_file in image_files:
            filename = os.path.basename(img_file)
            video_id = extract_video_id_from_filename(filename)

            if video_id:
                if filename.startswith('faceswap_'):
                    grouped_images[video_id]['faceswap'].append(img_file)
                elif filename.startswith('neuraltextures_'):
                    grouped_images[video_id]['neuraltextures'].append(img_file)

    # Get real test images
    real_test_dir = os.path.join(test_dir, "n01440764")
    if os.path.exists(real_test_dir):
        real_files = glob.glob(os.path.join(real_test_dir, "*.png")) + \
                     glob.glob(os.path.join(real_test_dir, "*.jpg"))

        for img_file in real_files:
            filename = os.path.basename(img_file)
            # Extract video ID from real images (format: real_videoname_frame.png)
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0] == 'real':
                video_id = parts[1]
                grouped_images[video_id]['real'].append(img_file)

    return grouped_images


def evaluate_paired_samples(model_path, grouped_images, device):
    """Evaluate model on paired samples"""
    
    # Load model
    model = CLIPLinearModel()

    # Load only classifier weights
    classifier_weights = torch.load(model_path, map_location=device)
    model.classifier.load_state_dict(classifier_weights)

    model.eval()
    model.to(device)
    
    # Define transforms
    tfms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    results = {
        'faceswap': {'predictions': [], 'probabilities': [], 'video_ids': [], 'labels': []},
        'neuraltextures': {'predictions': [], 'probabilities': [], 'video_ids': [], 'labels': []},
        'real': {'predictions': [], 'probabilities': [], 'video_ids': [], 'labels': []}
    }

    print("Evaluating model on paired samples...")

    for video_id, images in tqdm(grouped_images.items()):
        for method in ['faceswap', 'neuraltextures', 'real']:
            for img_path in images[method]:
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = img.convert('RGB')
                    img = tfms(img)

                    with torch.no_grad():
                        outputs = model(img.unsqueeze(0).to(device))
                        probs = torch.softmax(outputs, dim=1)

                        pred = torch.argmax(outputs, dim=1).item()
                        prob_fake = probs[0, 1].item()

                        # Label: 0 for real, 1 for fake
                        true_label = 0 if method == 'real' else 1

                        results[method]['predictions'].append(pred)
                        results[method]['probabilities'].append(prob_fake)
                        results[method]['video_ids'].append(video_id)
                        results[method]['labels'].append(true_label)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    true_label = 0 if method == 'real' else 1
                    results[method]['predictions'].append(1)
                    results[method]['probabilities'].append(0.5)
                    results[method]['video_ids'].append(video_id)
                    results[method]['labels'].append(true_label)
    
    return results


def calculate_metrics(results):
    """Calculate metrics for paired comparison"""
    metrics = {}

    for method in ['faceswap', 'neuraltextures', 'real']:
        if len(results[method]['predictions']) == 0:
            continue

        y_true = results[method]['labels']
        y_pred = results[method]['predictions']
        y_prob = results[method]['probabilities']

        print(f"\n{method} predictions summary:")
        print(f"  Samples: {len(y_true)}")
        print(f"  True labels: {set(y_true)}")
        print(f"  Unique predictions: {set(y_pred)}")
        print(f"  Probability range: {min(y_prob):.3f} - {max(y_prob):.3f}")
        print(f"  Mean probability: {np.mean(y_prob):.3f}")

        metrics[method] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'sample_count': len(y_true)
        }

        # Only calculate AUC if we have both classes in true labels
        unique_true = set(y_true)
        if len(unique_true) > 1:
            metrics[method]['f1_score'] = f1_score(y_true, y_pred, average='macro')
            metrics[method]['auc'] = roc_auc_score(y_true, y_prob)

            # Calculate EER
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
            metrics[method]['eer'] = fpr[eer_idx]
        else:
            print(f"  Warning: Only one class in true labels for {method}")
            metrics[method]['f1_score'] = accuracy_score(y_true, y_pred)  # For single class
            metrics[method]['auc'] = 0.5  # Random performance
            metrics[method]['eer'] = 0.5

    return metrics


def create_comparison_plot(metrics, output_dir):
    """Create comparison visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Paired Comparison: FaceSwap vs NeuralTextures', fontsize=14, fontweight='bold')
    
    methods = ['faceswap', 'neuraltextures']
    colors = ['#2E86AB', '#A23B72']
    
    # Accuracy
    ax1 = axes[0, 0]
    accuracies = [metrics[method]['accuracy'] for method in methods]
    bars1 = ax1.bar(methods, accuracies, color=colors)
    ax1.set_title('Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # AUC
    ax2 = axes[0, 1]
    aucs = [metrics[method]['auc'] for method in methods]
    bars2 = ax2.bar(methods, aucs, color=colors)
    ax2.set_title('AUC')
    ax2.set_ylabel('AUC')
    ax2.set_ylim(0, 1)
    for bar, auc in zip(bars2, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom')
    
    # F1 Score
    ax3 = axes[1, 0]
    f1s = [metrics[method]['f1_score'] for method in methods]
    bars3 = ax3.bar(methods, f1s, color=colors)
    ax3.set_title('F1 Score')
    ax3.set_ylabel('F1 Score')
    ax3.set_ylim(0, 1)
    for bar, f1 in zip(bars3, f1s):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{f1:.3f}', ha='center', va='bottom')
    
    # Sample count
    ax4 = axes[1, 1]
    counts = [metrics[method]['sample_count'] for method in methods]
    bars4 = ax4.bar(methods, counts, color=colors)
    ax4.set_title('Sample Count')
    ax4.set_ylabel('Number of Samples')
    for bar, count in zip(bars4, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paired_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Simple Paired Comparison Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset_path", type=str, default="../CLIPping-the-Deception/data", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="../results/paired_comparison", help="Output directory")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load split information
    data_dir = os.path.join(args.dataset_path, "ff_cross_forgery")
    split_info = load_split_info(data_dir)
    
    if not split_info:
        print("ERROR: Split information not found")
        return
    
    paired_videos = split_info.get('paired_videos', [])
    print(f"Found {len(paired_videos)} paired videos")
    
    # Group test images
    test_dir = os.path.join(data_dir, "images", "test")
    grouped_images = group_test_images_by_video(test_dir)
    
    # Filter to paired videos (with any combination of the three types)
    paired_grouped = {vid: imgs for vid, imgs in grouped_images.items()
                     if vid in paired_videos and
                     (len(imgs['faceswap']) > 0 or len(imgs['neuraltextures']) > 0 or len(imgs['real']) > 0)}

    print(f"Found {len(paired_grouped)} videos with samples")

    # Count videos with each type
    fs_count = sum(1 for imgs in paired_grouped.values() if len(imgs['faceswap']) > 0)
    nt_count = sum(1 for imgs in paired_grouped.values() if len(imgs['neuraltextures']) > 0)
    real_count = sum(1 for imgs in paired_grouped.values() if len(imgs['real']) > 0)

    print(f"  - FaceSwap: {fs_count} videos")
    print(f"  - NeuralTextures: {nt_count} videos")
    print(f"  - Real: {real_count} videos")
    
    if not paired_grouped:
        print("ERROR: No paired samples found")
        return
    
    # Evaluate
    results = evaluate_paired_samples(args.model_path, paired_grouped, device)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print("\n" + "="*60)
    print("CROSS-FORGERY EVALUATION RESULTS")
    print("="*60)

    for method in ['real', 'faceswap', 'neuraltextures']:
        if method in metrics:
            print(f"\n{method.upper()}:")
            print(f"  Samples: {metrics[method]['sample_count']}")
            print(f"  Accuracy: {metrics[method]['accuracy']:.4f}")
            print(f"  F1 Score: {metrics[method]['f1_score']:.4f}")
            print(f"  AUC: {metrics[method]['auc']:.4f}")
            print(f"  EER: {metrics[method]['eer']:.4f}")

    # Performance difference between fake types
    if 'faceswap' in metrics and 'neuraltextures' in metrics:
        acc_diff = metrics['faceswap']['accuracy'] - metrics['neuraltextures']['accuracy']
        auc_diff = metrics['faceswap']['auc'] - metrics['neuraltextures']['auc']

        print(f"\nCross-Forgery Generalization (FaceSwap vs NeuralTextures):")
        print(f"  Accuracy difference: {acc_diff:+.4f}")
        print(f"  AUC difference: {auc_diff:+.4f}")

        if acc_diff > 0.05:
            print(f"  → Model shows domain bias toward training data (FaceSwap)")
        elif acc_diff < -0.05:
            print(f"  → Model generalizes better to unseen domain (NeuralTextures)")
        else:
            print(f"  → Model shows balanced cross-domain performance")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'paired_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'video_count': len(paired_grouped)
        }, f, indent=2)
    
    # Create visualization
    create_comparison_plot(metrics, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
