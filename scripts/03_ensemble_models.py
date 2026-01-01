"""
QUICK WIN TRIO - PART 3: ADVANCED ENSEMBLE MODELS
=================================================
Building stacked ensemble, calibrated models, and optimized configurations
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, precision_recall_curve)
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("PART 3: ADVANCED ENSEMBLE MODELS")
print("="*80)

# Load data and selected features
print("\n[LOADING DATA]")
X_all = pd.read_csv('/home/claude/X_scaled.csv')
y = np.load('/home/claude/y.npy')

# Load feature sets
with open('/home/claude/selected_features.json', 'r') as f:
    feature_sets = json.load(f)

print(f"✓ Total features available: {X_all.shape[1]}")
print(f"✓ Samples: {X_all.shape[0]}")

# Use best performing feature set from previous step + consensus features
original_paper_proteins = ['KRT19', 'HAOX1', 'CXCL1', 'RARRES2', 'FCRL6', 'REN', 'IL13',
                           'SPON1', 'MMP-1', 'ARNT', 'TNFSF13B', 'PRKCQ', 'TRAIL-R2',
                           'hOSCAR', 'MCP-2', 'DPP10', 'GDNF', 'male', 'BLDAS']

# Combine original + engineered consensus features
best_features = list(set(original_paper_proteins + feature_sets['consensus'][:20]))
best_features = [f for f in best_features if f in X_all.columns]

X = X_all[best_features].copy()

print(f"\n✓ Using {len(best_features)} features for modeling")
print(f"✓ Feature types:")
print(f"  - Original paper proteins: {len([f for f in best_features if f in original_paper_proteins])}")
print(f"  - Engineered features: {len([f for f in best_features if f not in original_paper_proteins])}")

# ============================================================================
# DEFINE BASE MODELS
# ============================================================================
print("\n" + "="*80)
print("DEFINING BASE MODELS")
print("="*80)

base_models = {}

# Model 1: Logistic Regression (Balanced)
base_models['LogReg_Balanced'] = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    solver='liblinear',
    C=1.0,
    random_state=RANDOM_STATE
)

# Model 2: Elastic Net (Original Paper)
base_models['ElasticNet'] = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.9,  # 90% LASSO, 10% Ridge (original paper)
    class_weight='balanced',
    max_iter=2000,
    C=1.0,
    random_state=RANDOM_STATE
)

# Model 3: Random Forest (Controlled)
base_models['RandomForest'] = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Model 4: Gradient Boosting (Controlled)
base_models['GradientBoosting'] = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=RANDOM_STATE
)

# Model 5: SVM with RBF kernel
base_models['SVM_RBF'] = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=RANDOM_STATE
)

print(f"✓ Defined {len(base_models)} base models:")
for name in base_models.keys():
    print(f"  - {name}")

# ============================================================================
# EVALUATE BASE MODELS
# ============================================================================
print("\n" + "="*80)
print("EVALUATING BASE MODELS")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

base_results = []

print("\nRunning 5-fold cross-validation for each model...")

for name, model in base_models.items():
    print(f"\n→ Evaluating {name}...")
    
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'roc_auc', 'precision', 'recall'],
        return_train_score=True,
        n_jobs=-1
    )
    
    result = {
        'Model': name,
        'Train_AUC': cv_results['train_roc_auc'].mean(),
        'Test_AUC': cv_results['test_roc_auc'].mean(),
        'Test_AUC_Std': cv_results['test_roc_auc'].std(),
        'Test_Accuracy': cv_results['test_accuracy'].mean(),
        'Test_Accuracy_Std': cv_results['test_accuracy'].std(),
        'Test_Precision': cv_results['test_precision'].mean(),
        'Test_Recall': cv_results['test_recall'].mean(),
        'Overfitting': cv_results['train_roc_auc'].mean() - cv_results['test_roc_auc'].mean()
    }
    
    base_results.append(result)
    
    print(f"  AUC: {result['Test_AUC']:.4f} ± {result['Test_AUC_Std']:.4f}")
    print(f"  Accuracy: {result['Test_Accuracy']:.4f} ± {result['Test_Accuracy_Std']:.4f}")
    print(f"  Overfitting: {result['Overfitting']:.4f}")

base_results_df = pd.DataFrame(base_results).sort_values('Test_AUC', ascending=False)

print("\n" + "-"*80)
print("BASE MODEL RESULTS")
print("-"*80)
print(base_results_df.to_string(index=False))

# ============================================================================
# VOTING ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("VOTING ENSEMBLE")
print("="*80)

# Soft voting (weighted by performance)
voting_clf = VotingClassifier(
    estimators=[
        ('lr', base_models['LogReg_Balanced']),
        ('en', base_models['ElasticNet']),
        ('rf', base_models['RandomForest']),
        ('gb', base_models['GradientBoosting'])
    ],
    voting='soft',
    weights=[2, 2, 1, 1]  # Weight better performers higher
)

print("\n→ Evaluating Voting Ensemble...")

voting_results = cross_validate(
    voting_clf, X, y, cv=cv,
    scoring=['accuracy', 'roc_auc', 'precision', 'recall'],
    return_train_score=True,
    n_jobs=-1
)

print(f"  AUC: {voting_results['test_roc_auc'].mean():.4f} ± {voting_results['test_roc_auc'].std():.4f}")
print(f"  Accuracy: {voting_results['test_accuracy'].mean():.4f} ± {voting_results['test_accuracy'].std():.4f}")

# ============================================================================
# STACKING ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("STACKING ENSEMBLE (META-LEARNING)")
print("="*80)

# Level 0: Base estimators
stacking_estimators = [
    ('lr', base_models['LogReg_Balanced']),
    ('en', base_models['ElasticNet']),
    ('rf', base_models['RandomForest']),
    ('gb', base_models['GradientBoosting']),
    ('svm', base_models['SVM_RBF'])
]

# Level 1: Meta-learner (Logistic Regression with L2)
meta_learner = LogisticRegression(
    C=1.0,
    random_state=RANDOM_STATE,
    max_iter=1000
)

stacking_clf = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

print("\n→ Evaluating Stacking Ensemble...")

stacking_results = cross_validate(
    stacking_clf, X, y, cv=cv,
    scoring=['accuracy', 'roc_auc', 'precision', 'recall'],
    return_train_score=True,
    n_jobs=-1
)

print(f"  AUC: {stacking_results['test_roc_auc'].mean():.4f} ± {stacking_results['test_roc_auc'].std():.4f}")
print(f"  Accuracy: {stacking_results['test_accuracy'].mean():.4f} ± {stacking_results['test_accuracy'].std():.4f}")

# ============================================================================
# CALIBRATED MODELS
# ============================================================================
print("\n" + "="*80)
print("CALIBRATED MODELS")
print("="*80)

# Calibrate the best base model
best_base_model = base_results_df.iloc[0]['Model']
print(f"\n→ Calibrating best base model: {best_base_model}")

calibrated_clf = CalibratedClassifierCV(
    base_models[best_base_model],
    method='sigmoid',
    cv=5
)

calibrated_results = cross_validate(
    calibrated_clf, X, y, cv=cv,
    scoring=['accuracy', 'roc_auc', 'precision', 'recall'],
    return_train_score=True,
    n_jobs=-1
)

print(f"  AUC: {calibrated_results['test_roc_auc'].mean():.4f} ± {calibrated_results['test_roc_auc'].std():.4f}")
print(f"  Accuracy: {calibrated_results['test_accuracy'].mean():.4f} ± {calibrated_results['test_accuracy'].std():.4f}")

# ============================================================================
# FINAL MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

final_results = base_results.copy()

# Add ensemble results
final_results.append({
    'Model': 'Voting Ensemble',
    'Train_AUC': voting_results['train_roc_auc'].mean(),
    'Test_AUC': voting_results['test_roc_auc'].mean(),
    'Test_AUC_Std': voting_results['test_roc_auc'].std(),
    'Test_Accuracy': voting_results['test_accuracy'].mean(),
    'Test_Accuracy_Std': voting_results['test_accuracy'].std(),
    'Test_Precision': voting_results['test_precision'].mean(),
    'Test_Recall': voting_results['test_recall'].mean(),
    'Overfitting': voting_results['train_roc_auc'].mean() - voting_results['test_roc_auc'].mean()
})

final_results.append({
    'Model': 'Stacking Ensemble',
    'Train_AUC': stacking_results['train_roc_auc'].mean(),
    'Test_AUC': stacking_results['test_roc_auc'].mean(),
    'Test_AUC_Std': stacking_results['test_roc_auc'].std(),
    'Test_Accuracy': stacking_results['test_accuracy'].mean(),
    'Test_Accuracy_Std': stacking_results['test_accuracy'].std(),
    'Test_Precision': stacking_results['test_precision'].mean(),
    'Test_Recall': stacking_results['test_recall'].mean(),
    'Overfitting': stacking_results['train_roc_auc'].mean() - stacking_results['test_roc_auc'].mean()
})

final_results.append({
    'Model': f'Calibrated {best_base_model}',
    'Train_AUC': calibrated_results['train_roc_auc'].mean(),
    'Test_AUC': calibrated_results['test_roc_auc'].mean(),
    'Test_AUC_Std': calibrated_results['test_roc_auc'].std(),
    'Test_Accuracy': calibrated_results['test_accuracy'].mean(),
    'Test_Accuracy_Std': calibrated_results['test_accuracy'].std(),
    'Test_Precision': calibrated_results['test_precision'].mean(),
    'Test_Recall': calibrated_results['test_recall'].mean(),
    'Overfitting': calibrated_results['train_roc_auc'].mean() - calibrated_results['test_roc_auc'].mean()
})

final_results_df = pd.DataFrame(final_results).sort_values('Test_AUC', ascending=False)

print("\n" + "="*80)
print("ALL MODELS - RANKED BY TEST AUC")
print("="*80)
print(f"\n{'Model':<30} {'Test AUC':<20} {'Test Accuracy':<20} {'Overfitting':<15}")
print("-"*85)
for _, row in final_results_df.iterrows():
    print(f"{row['Model']:<30} {row['Test_AUC']:.4f}±{row['Test_AUC_Std']:.4f}       "
          f"{row['Test_Accuracy']:.4f}±{row['Test_Accuracy_Std']:.4f}       "
          f"{row['Overfitting']:.4f}")

# Save results
final_results_df.to_csv('/home/claude/ensemble_results.csv', index=False)
print("\n✓ Saved results to: ensemble_results.csv")

# ============================================================================
# COMPARISON WITH BASELINE
# ============================================================================
print("\n" + "="*80)
print("COMPARISON WITH PUBLISHED RESULTS")
print("="*80)

best_model = final_results_df.iloc[0]

print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"\n{'Metric':<30} {'Original ATRPred':<20} {'Your Article':<20} {'Our Result':<20}")
print("-"*90)

acc_pct = f"{best_model['Test_Accuracy']*100:.1f}%"
auc_val = f"{best_model['Test_AUC']:.3f}"
recall_pct = f"{best_model['Test_Recall']*100:.1f}%"
precision_pct = f"{best_model['Test_Precision']*100:.1f}%"

print(f"{'Accuracy':<30} {'81.0%':<20} {'87.5%':<20} {acc_pct:<20}")
print(f"{'AUC':<30} {'0.860':<20} {'0.917':<20} {auc_val:<20}")
print(f"{'Sensitivity (Recall)':<30} {'75.0%':<20} {'91.7%':<20} {recall_pct:<20}")
print(f"{'Precision':<30} {'-':<20} {'-':<20} {precision_pct:<20}")

# Calculate specificity from precision and recall (if needed)
# Specificity = TN / (TN + FP)
# For class imbalance, we can estimate

print("\n" + "="*80)
print("IMPROVEMENTS ACHIEVED")
print("="*80)

improvements = {
    'vs_original': {
        'accuracy': (best_model['Test_Accuracy'] - 0.81) * 100,
        'auc': (best_model['Test_AUC'] - 0.86) * 100
    },
    'vs_article': {
        'accuracy': (best_model['Test_Accuracy'] - 0.875) * 100,
        'auc': (best_model['Test_AUC'] - 0.917) * 100
    }
}

print(f"\nVs. Original ATRPred:")
print(f"  Accuracy: {improvements['vs_original']['accuracy']:+.1f}%")
print(f"  AUC: {improvements['vs_original']['auc']:+.1f}%")

print(f"\nVs. Your Article:")
print(f"  Accuracy: {improvements['vs_article']['accuracy']:+.1f}%")
print(f"  AUC: {improvements['vs_article']['auc']:+.1f}%")

print("\n" + "="*80)
print("ENSEMBLE MODELING COMPLETE")
print("="*80)
