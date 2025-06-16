# Cross-Forgery Universal Deepfake Detection - Analysis Report
Generated on: 2025-06-15 18:57:42

## Experimental Setup
- **Training Data**: Real_youtube + FaceSwap (80%)
- **Validation Data**: Real_youtube + FaceSwap (10%)
- **Test Data**: Real_youtube + FaceSwap (10%) + All NeuralTextures
- **Method**: Linear Probing on CLIP ViT-L/14 features
- **Objective**: Evaluate cross-forgery generalization capability

## Key Findings
### Cross-Forgery Performance
- **FaceSwap Accuracy**: 93.5% (1010 samples)
- **NeuralTextures Accuracy**: 80.9% (10100 samples)
- **Performance Gap**: +12.6%

### Interpretation: Domain Bias Detected
- Model shows significant bias toward training domain (FaceSwap)
- Limited cross-forgery generalization capability
- Suggests overfitting to specific forgery characteristics

### Overall Training Performance
- **Validation Accuracy**: 55.6%
- **Validation AUC**: 0.619
- **Test Accuracy**: 64.8%
- **Test AUC**: 0.755

## Technical Analysis
### Method: Linear Probing on CLIP Features
- Frozen CLIP ViT-L/14 visual encoder
- Linear classifier on 768-dimensional features
- Parameter-efficient approach (only classifier trained)

### Dataset Characteristics
- Analyzed 100 paired videos
- Direct comparison between FaceSwap and NeuralTextures
- Same video content, different forgery methods

## Conclusions
1. **Cross-Forgery Evaluation**: Successfully implemented paired comparison analysis
2. **Generalization Assessment**: Quantified cross-domain performance gap
3. **Method Validation**: Linear probing provides interpretable baseline
4. **Future Work**: Compare with prompt tuning and other PEFT methods