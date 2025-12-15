# Bone Age Prediction from Hand Radiographs

Automated bone age estimation from pediatric hand X-rays using deep learning. Implements both regression (continuous age prediction) and classification (developmental staging) approaches.


*PRML Course Project, IIITDM Kancheepuram*

## 🎯 Key Results

### Regression: Predicting Bone Age in Years
- **Best Model**: Optimized End-to-End CNN
- **MAE**: 0.7083 years (8.5 months) - *below radiologist variability*
- **R²**: 0.9246
- **Improvement**: 35% better than best hybrid model

### Classification: Developmental Staging (5 Classes)
- **Best Model**: ResNet-50 CNN Classifier  
- **QWK**: 0.8888 (near-expert agreement)
- **Accuracy**: 78.17%
- **Improvement**: 77% better than HOG+XGBoost baseline

## 📊 Dataset

**RSNA Pediatric Bone Age Dataset** (Kaggle)
- 12,611 hand radiographs with bone age labels and sex
- Split: 70% train / 15% validation / 15% test

## 🏗️ Methodology

### Regression Task
1. **Hybrid CNN + Classical ML**: ResNet-50 features → Ridge/SVR/RF/XGBoost
2. **Baseline End-to-End CNN**: Fine-tuned ResNet-50 with simple head
3. **Optimized CNN** 🏆: Enhanced head + MAE loss + regularization

### Classification Task
1. **CNN Classifier** 🏆: ResNet-50 fine-tuned for 5-class staging
2. **HOG + XGBoost**: Traditional computer vision baseline

**Developmental Stages**: Early Childhood (0-4yr) | Mid Childhood (4-8yr) | Pre-Adolescence (8-12yr) | Adolescence (12-15yr) | Mature (15+yr)

## 📈 Performance Comparison

### Regression Results
| Model | MAE (Years) | RMSE | R² |
|-------|-------------|------|-----|
| **Optimized CNN** 🏆 | **0.7083** | **0.9357** | **0.9246** |
| Ridge Regression | 1.0767 | 1.4030 | 0.8250 |
| SVR | 1.0958 | 1.4246 | 0.8180 |
| XGBoost | 1.3029 | 1.7174 | 0.7367 |

### Classification Results
| Model | QWK | Accuracy |
|-------|-----|----------|
| **ResNet-50 CNN** 🏆 | **0.8888** | **78.17%** |
| HOG + XGBoost | 0.5016 | 47.62% |

## 📁 Repository Structure

```
Boneage/
├── regress_optimized.ipynb          # Optimized regression model
├── regress.ipynb                    # Baseline regression
├── classification.ipynb             # Classification model
├── gradcam.ipynb                    # Visualization
├── boneage-training-dataset.csv     # Training metadata
├── boneage-test-dataset.csv         # Test metadata
└── Documentation_PRML.pdf           # Full documentation
```

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/harshinis30/Boneage.git
cd Boneage

# Install dependencies
pip install torch torchvision pandas numpy scikit-learn matplotlib xgboost

# Download dataset from Kaggle
# https://www.kaggle.com/c/rsna-bone-age

# Run notebooks
jupyter notebook
```

## 🔬 Technical Details

**Preprocessing**: Grayscale → 256×256 resize → Normalization (μ=0.5, σ=0.5)  
**Augmentation**: Random flip, rotation (±20°), affine transforms, brightness/contrast jitter  
**Architecture**: ResNet-50 backbone + custom regression/classification heads  
**Optimization**: Adam optimizer, ReduceLROnPlateau scheduler, early stopping

## 🎓 Key Findings

- End-to-end fine-tuning significantly outperforms frozen feature extraction
- Deep learning models achieve clinically acceptable error rates
- CNN features show strong linear separability for regression
- Classification errors occur primarily between adjacent developmental stages
- Minimal gender bias (Male: 78.68%, Female: 77.57%)

## 🔮 Future Work

- Transformer-based architectures (Vision Transformers)
- Multi-task learning (joint regression + classification)
- Cross-dataset validation
- Attention mechanisms for interpretability

## 📚 References

1. [RSNA Pediatric Bone Age Dataset](https://www.kaggle.com/c/rsna-bone-age)
2. He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
3. Greulich & Pyle (1959). Radiographic Atlas of Skeletal Development.


---

*See [Documentation_PRML.pdf](Documentation_PRML.pdf) for detailed methodology and analysis.*
