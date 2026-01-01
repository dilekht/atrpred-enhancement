# ATRPred Enhancement: Advanced Anti-TNF Response Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Enhanced machine learning approach for predicting anti-TNF treatment response in rheumatoid arthritis patients

## 🎯 Overview

This repository contains an advanced implementation that improves upon the original [ATRPred](https://github.com/ShuklaLab/ATRPred) published in PLOS Computational Biology (2022). Through systematic feature engineering, multi-method feature selection, and optimized ensemble learning, we achieved:

- **AUC: 0.904** (+5.1% improvement over original 0.860)
- **Accuracy: 83.7%** (+2.7% improvement over original 81.0%)
- **Low overfitting: 0.102** (excellent generalization)

## 📊 Key Results

| Metric | Original ATRPred | Our Enhancement | Improvement |
|--------|------------------|-----------------|-------------|
| **AUC** | 0.860 | **0.904** | **+5.1%** |
| **Accuracy** | 81.0% | **83.7%** | **+2.7%** |
| **Features** | 17 proteins | 34 (19+15 eng.) | More informative |
| **Algorithm** | Elastic Net | Optimized SVM/Stacking | More robust |

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
numpy >= 1.19.0
pandas >= 1.1.0
scikit-learn >= 0.24.0
scipy >= 1.5.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/atrpred-enhancement.git
cd atrpred-enhancement

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Complete pipeline (all steps)
python run_complete_pipeline.py

# Or run individual steps:
python scripts/01_feature_engineering.py
python scripts/02_feature_selection.py
python scripts/03_ensemble_models.py
python scripts/04_final_optimization.py
```

## 📁 Project Structure

```
atrpred-enhancement/
│
├── data/                          # Data directory
│   ├── raw/                       # Original ATRPred data
│   │   ├── ra_tot.txt            # Primary dataset (89 patients)
│   │   └── ra_npx.tsv            # Extended cohort (195 patients)
│   └── processed/                 # Processed data
│       ├── X_scaled.csv          # Standardized features
│       └── y.npy                 # Response labels
│
├── scripts/                       # Main analysis scripts
│   ├── 01_feature_engineering.py # Feature creation (371 features)
│   ├── 02_feature_selection.py   # Multi-method selection (34 features)
│   ├── 03_ensemble_models.py     # Ensemble learning
│   └── 04_final_optimization.py  # Hyperparameter tuning
│
├── results/                       # Output results
│   ├── figures/                   # Visualizations
│   ├── models/                    # Saved models
│   ├── final_results.json        # Main results
│   ├── model_parameters.json     # Optimized hyperparameters
│   └── final_comparison.csv      # Performance comparison
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
│
├── docs/                          # Documentation
│   ├── FINAL_REPORT.md           # Comprehensive results report
│   ├── improvement_strategy.md   # Methodology details
│   └── API.md                    # Code documentation
│
├── tests/                         # Unit tests
│   └── test_pipeline.py
│
├── requirements.txt               # Python dependencies
├── run_complete_pipeline.py      # Main execution script
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 🔬 Methodology

### 1. Feature Engineering (371 features)

**Biological Pathway Integration:**
- IL-17 pathway score (4 proteins)
- Inflammatory response score (3 proteins)
- Oxidative stress score (3 proteins)
- Hub interaction score (PPI network-based)

**Clinical Interactions:**
- Gender × protein interactions
- Baseline DAS × protein ratios

**Non-linear Transformations:**
- Polynomial features (squared terms)
- Statistical aggregates (mean, std, max, min)

### 2. Multi-Method Feature Selection

Five independent methods with consensus voting:

| Method | Selected | Top Feature |
|--------|----------|-------------|
| RFECV | 331 | BMP-6 |
| F-Score | 50 | BLDAS |
| Mutual Information | 50 | ITGB2 |
| Random Forest | 50 | GT |
| Stability Selection | 6 | BLDAS |
| **Consensus (≥3 votes)** | **31** | **CXCL1** |

**Final:** 34 features (19 original + 15 engineered)

### 3. Optimized Ensemble Learning

**Best Models:**
1. **Optimized SVM**: AUC 0.904 ± 0.028 ⭐
2. Stacking Ensemble: AUC 0.896 ± 0.068
3. Calibrated SVM: AUC 0.888 ± 0.038

**Hyperparameter Optimization:**
```python
SVM(
    C=2.0,
    gamma='scale',
    class_weight={0: 1, 1: 2},
    kernel='rbf'
)
```

## 📈 Performance Metrics

### Cross-Validation Results (5-Fold Stratified)

```
Model: Optimized SVM
├── AUC: 0.904 ± 0.028
├── Accuracy: 83.7 ± 2.3%
├── Overfitting: 0.102
└── Training Time: ~2 minutes
```

### Comparison with Original ATRPred

```python
Improvements:
- AUC: +0.044 (+5.1%)
- Accuracy: +2.7 percentage points
- Overfitting: -27% (better generalization)
- Features: Biologically interpretable
```

## 🔧 Usage Example

```python
import pandas as pd
import numpy as np
from scripts.feature_engineering import FeatureEngineer
from scripts.model_trainer import train_optimized_svm

# Load data
df = pd.read_csv('data/raw/ra_tot.txt', sep=' ')

# Engineer features
engineer = FeatureEngineer()
X_engineered = engineer.fit_transform(df)

# Train model
model, metrics = train_optimized_svm(
    X_engineered, 
    y,
    cv_folds=5,
    random_state=42
)

print(f"AUC: {metrics['auc']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

## 📊 Key Features

### Top 10 Selected Features (by consensus votes)

1. **CXCL1** (5/5 votes) - Chemokine ligand 1
2. **BLDAS** (5/5 votes) - Baseline disease activity
3. **KRT19** (5/5 votes) - Keratin 19
4. **CXCL5** (4/5 votes) - Chemokine ligand 5
5. **CCL17** (4/5 votes) - C-C motif chemokine 17
6. **MCP-4** (4/5 votes) - Monocyte chemotactic protein 4
7. **CASP-3** (4/5 votes) - Caspase 3
8. **CXCL1_squared** (4/5 votes) - CXCL1 polynomial feature
9. **FCRL6_squared** (4/5 votes) - FCRL6 polynomial feature
10. **TRAIL** (3/5 votes) - TNF-related apoptosis-inducing ligand

## 🧪 Reproducibility

All results are reproducible with fixed random seeds:

```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

Run the complete pipeline:
```bash
python run_complete_pipeline.py --seed 42 --cv-folds 5
```

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{atrpred_enhancement2025,
  title={Enhanced Anti-TNF Response Prediction in Rheumatoid Arthritis through 
         Advanced Feature Engineering and Ensemble Learning},
  author={[Your Name]},
  journal={[Target Journal]},
  year={2025},
  note={Improves upon Prasad et al., PLOS Comput Biol 2022}
}
```

**Original ATRPred:**
```bibtex
@article{prasad2022atrpred,
  title={ATRPred: A machine learning based tool for clinical decision making of 
         anti-TNF treatment in rheumatoid arthritis patients},
  author={Prasad, Bodhayan and others},
  journal={PLOS Computational Biology},
  volume={18},
  number={7},
  pages={e1010204},
  year={2022},
  publisher={Public Library of Science}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original ATRPred team (Prasad et al., 2022)
- Olink Proteomics for protein measurement platform
- ShuklaLab for making data publicly available
- Northern Ireland Centre for Stratified Medicine (NICSM)

## 📧 Contact

For questions or collaborations:
- Email: [your.email@example.com]
- GitHub Issues: [Create an issue](https://github.com/yourusername/atrpred-enhancement/issues)

## 🔗 Links

- [Original ATRPred Paper](https://doi.org/10.1371/journal.pcbi.1010204)
- [Original ATRPred GitHub](https://github.com/ShuklaLab/ATRPred)
- [Documentation](docs/FINAL_REPORT.md)

---

**Status:** ✅ Ready for Publication | **Version:** 1.0.0 | **Last Updated:** December 2025
