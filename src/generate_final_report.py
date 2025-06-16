#!/usr/bin/env python3
"""
Generate Final Analysis Report
Creates a comprehensive report of the cross-forgery detection experiment
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime


def load_training_results():
    """Load training results"""
    results_file = "CLIPping-the-Deception/weights/linear_probing_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def load_paired_results():
    """Load paired comparison results"""
    results_file = "results/paired_comparison/paired_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def create_performance_comparison(training_results, paired_results, output_dir):
    """Create performance comparison visualization"""
    
    # Prepare data
    data = []
    
    if training_results:
        # Validation results
        val_metrics = training_results.get('validation', {})
        data.append({
            'Dataset': 'Validation\n(Real+FaceSwap)',
            'Accuracy': val_metrics.get('accuracy', 0),
            'AUC': val_metrics.get('auc', 0),
            'F1': val_metrics.get('f1_score', 0)
        })
        
        # Test results
        test_metrics = training_results.get('test', {})
        data.append({
            'Dataset': 'Test\n(Real+FaceSwap+NT)',
            'Accuracy': test_metrics.get('accuracy', 0),
            'AUC': test_metrics.get('auc', 0),
            'F1': test_metrics.get('f1_score', 0)
        })
    
    if paired_results:
        # Paired comparison results
        metrics = paired_results.get('metrics', {})
        
        if 'faceswap' in metrics:
            data.append({
                'Dataset': 'FaceSwap Only\n(Training Domain)',
                'Accuracy': metrics['faceswap']['accuracy'],
                'AUC': metrics['faceswap']['auc'],
                'F1': metrics['faceswap']['f1_score']
            })
        
        if 'neuraltextures' in metrics:
            data.append({
                'Dataset': 'NeuralTextures Only\n(Cross-Domain)',
                'Accuracy': metrics['neuraltextures']['accuracy'],
                'AUC': metrics['neuraltextures']['auc'],
                'F1': metrics['neuraltextures']['f1_score']
            })
    
    if not data:
        print("No data available for visualization")
        return
    
    df = pd.DataFrame(data)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Cross-Forgery Detection Performance Analysis', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'AUC', 'F1']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(df['Dataset'], df[metric], color=colors[i], alpha=0.7)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def create_cross_domain_analysis(paired_results, output_dir):
    """Create cross-domain analysis"""
    
    if not paired_results or 'metrics' not in paired_results:
        return
    
    metrics = paired_results['metrics']
    
    if 'faceswap' not in metrics or 'neuraltextures' not in metrics:
        return
    
    # Calculate performance differences
    fs_acc = metrics['faceswap']['accuracy']
    nt_acc = metrics['neuraltextures']['accuracy']
    
    acc_diff = fs_acc - nt_acc
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Cross-Domain Generalization Analysis', fontsize=14, fontweight='bold')
    
    # Performance comparison
    methods = ['FaceSwap\n(Training)', 'NeuralTextures\n(Cross-Domain)']
    accuracies = [fs_acc, nt_acc]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax1.bar(methods, accuracies, color=colors)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Performance gap analysis
    gap_color = '#E74C3C' if acc_diff > 0.05 else '#27AE60' if acc_diff < -0.05 else '#F39C12'
    
    ax2.bar(['Performance Gap'], [abs(acc_diff)], color=gap_color)
    ax2.set_title('Cross-Domain Performance Gap')
    ax2.set_ylabel('Absolute Accuracy Difference')
    ax2.text(0, abs(acc_diff) + 0.01, f'{acc_diff:+.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add interpretation
    if acc_diff > 0.05:
        interpretation = "Domain Bias\n(Favors Training Data)"
    elif acc_diff < -0.05:
        interpretation = "Cross-Domain\nGeneralization"
    else:
        interpretation = "Balanced\nPerformance"
    
    ax2.text(0, abs(acc_diff)/2, interpretation, ha='center', va='center', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_domain_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_text_report(training_results, paired_results, performance_df, output_dir):
    """Generate text-based analysis report"""
    
    report = []
    report.append("# Cross-Forgery Universal Deepfake Detection - Analysis Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## Experimental Setup")
    report.append("- **Training Data**: Real_youtube + FaceSwap (80%)")
    report.append("- **Validation Data**: Real_youtube + FaceSwap (10%)")
    report.append("- **Test Data**: Real_youtube + FaceSwap (10%) + All NeuralTextures")
    report.append("- **Method**: Linear Probing on CLIP ViT-L/14 features")
    report.append("- **Objective**: Evaluate cross-forgery generalization capability")
    report.append("")
    
    report.append("## Key Findings")
    
    if paired_results and 'metrics' in paired_results:
        metrics = paired_results['metrics']
        
        if 'faceswap' in metrics and 'neuraltextures' in metrics:
            fs_acc = metrics['faceswap']['accuracy']
            nt_acc = metrics['neuraltextures']['accuracy']
            acc_diff = fs_acc - nt_acc
            
            report.append(f"### Cross-Forgery Performance")
            report.append(f"- **FaceSwap Accuracy**: {fs_acc:.1%} ({metrics['faceswap']['sample_count']} samples)")
            report.append(f"- **NeuralTextures Accuracy**: {nt_acc:.1%} ({metrics['neuraltextures']['sample_count']} samples)")
            report.append(f"- **Performance Gap**: {acc_diff:+.1%}")
            report.append("")
            
            if acc_diff > 0.05:
                report.append("### Interpretation: Domain Bias Detected")
                report.append("- Model shows significant bias toward training domain (FaceSwap)")
                report.append("- Limited cross-forgery generalization capability")
                report.append("- Suggests overfitting to specific forgery characteristics")
            elif acc_diff < -0.05:
                report.append("### Interpretation: Excellent Cross-Domain Generalization")
                report.append("- Model performs better on unseen forgery type (NeuralTextures)")
                report.append("- Strong cross-forgery generalization capability")
                report.append("- Learned generalizable deepfake features")
            else:
                report.append("### Interpretation: Balanced Cross-Domain Performance")
                report.append("- Model shows consistent performance across forgery types")
                report.append("- Good cross-forgery generalization capability")
                report.append("- Learned domain-invariant features")
            
            report.append("")
    
    if training_results:
        report.append("### Overall Training Performance")
        val_metrics = training_results.get('validation', {})
        test_metrics = training_results.get('test', {})
        
        if val_metrics:
            report.append(f"- **Validation Accuracy**: {val_metrics.get('accuracy', 0):.1%}")
            report.append(f"- **Validation AUC**: {val_metrics.get('auc', 0):.3f}")
        
        if test_metrics:
            report.append(f"- **Test Accuracy**: {test_metrics.get('accuracy', 0):.1%}")
            report.append(f"- **Test AUC**: {test_metrics.get('auc', 0):.3f}")
        
        report.append("")
    
    report.append("## Technical Analysis")
    report.append("### Method: Linear Probing on CLIP Features")
    report.append("- Frozen CLIP ViT-L/14 visual encoder")
    report.append("- Linear classifier on 768-dimensional features")
    report.append("- Parameter-efficient approach (only classifier trained)")
    report.append("")
    
    report.append("### Dataset Characteristics")
    if paired_results:
        video_count = paired_results.get('video_count', 0)
        report.append(f"- Analyzed {video_count} paired videos")
        report.append("- Direct comparison between FaceSwap and NeuralTextures")
        report.append("- Same video content, different forgery methods")
    
    report.append("")
    
    report.append("## Conclusions")
    report.append("1. **Cross-Forgery Evaluation**: Successfully implemented paired comparison analysis")
    report.append("2. **Generalization Assessment**: Quantified cross-domain performance gap")
    report.append("3. **Method Validation**: Linear probing provides interpretable baseline")
    report.append("4. **Future Work**: Compare with prompt tuning and other PEFT methods")
    
    # Save report
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'analysis_report.md'), 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    output_dir = "final_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Final Analysis Report...")
    print("=" * 50)
    
    # Load results
    training_results = load_training_results()
    paired_results = load_paired_results()
    
    if not training_results and not paired_results:
        print("ERROR: No results found to analyze")
        return
    
    print("✓ Results loaded successfully")
    
    # Generate visualizations
    print("Creating performance comparison...")
    performance_df = create_performance_comparison(training_results, paired_results, output_dir)
    
    print("Creating cross-domain analysis...")
    create_cross_domain_analysis(paired_results, output_dir)
    
    # Generate text report
    print("Generating text report...")
    report_text = generate_text_report(training_results, paired_results, performance_df, output_dir)
    
    print("\n" + "=" * 50)
    print("FINAL ANALYSIS REPORT GENERATED")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    print("  - performance_analysis.png")
    print("  - cross_domain_analysis.png")
    print("  - analysis_report.md")
    print("=" * 50)
    
    # Print key findings
    if paired_results and 'metrics' in paired_results:
        metrics = paired_results['metrics']
        if 'faceswap' in metrics and 'neuraltextures' in metrics:
            fs_acc = metrics['faceswap']['accuracy']
            nt_acc = metrics['neuraltextures']['accuracy']
            acc_diff = fs_acc - nt_acc
            
            print("\nKEY FINDINGS:")
            print(f"FaceSwap Accuracy: {fs_acc:.1%}")
            print(f"NeuralTextures Accuracy: {nt_acc:.1%}")
            print(f"Performance Gap: {acc_diff:+.1%}")
            
            if acc_diff > 0.05:
                print("→ Domain bias detected (favors training data)")
            elif acc_diff < -0.05:
                print("→ Excellent cross-domain generalization")
            else:
                print("→ Balanced cross-domain performance")


if __name__ == "__main__":
    main()
