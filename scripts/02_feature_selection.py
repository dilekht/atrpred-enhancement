"""
QUICK WIN TRIO - PART 2: ADVANCED FEATURE SELECTION
===================================================
Implementing RFECV, Statistical Selection, and Combined Approach
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("PART 2: ADVANCED FEATURE SELECTION")
print("="*80)

# Load preprocessed data
print("\n[LOADING DATA]")
X = pd.read_csv('/home/claude/X_scaled.csv')
y = np.load('/home/claude/y.npy')

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Samples: {X.shape[0]}")
print(f"✓ Responders: {y.sum()} / Non-responders: {(y==0).sum()}")

# ============================================================================
# METHOD 1: Recursive Feature Elimination with Cross-Validation (RFECV)
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: RECURSIVE FEATURE ELIMINATION (RFECV)")
print("="*80)

print("\nTraining RFECV with Logistic Regression...")
print("(This may take a few minutes...)")

# Use logistic regression as base estimator
log_reg = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    solver='liblinear',
    random_state=RANDOM_STATE
)

# RFECV with 5-fold stratified CV
rfecv = RFECV(
    estimator=log_reg,
    step=5,  # Remove 5 features at a time for speed
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc',
    min_features_to_select=15,
    n_jobs=-1
)

rfecv.fit(X, y)

rfecv_features = X.columns[rfecv.support_].tolist()
rfecv_scores = rfecv.cv_results_['mean_test_score']

print(f"\n✓ Optimal number of features: {rfecv.n_features_}")
print(f"✓ Best cross-validated AUC: {max(rfecv_scores):.4f}")
print(f"\nTop 20 selected features:")
for i, feat in enumerate(rfecv_features[:20], 1):
    print(f"  {i:2d}. {feat}")

# ============================================================================
# METHOD 2: Statistical Feature Selection (F-Score and Mutual Information)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: STATISTICAL FEATURE SELECTION")
print("="*80)

# F-score (ANOVA F-value)
print("\n→ Computing F-scores...")
selector_f = SelectKBest(score_func=f_classif, k=50)
selector_f.fit(X, y)

f_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': selector_f.scores_,
    'p_value': selector_f.pvalues_
}).sort_values('f_score', ascending=False)

f_features = f_scores.head(50)['feature'].tolist()

print(f"✓ Selected top 50 features by F-score")
print(f"\nTop 10 features by F-score:")
for i, row in f_scores.head(10).iterrows():
    print(f"  {row['feature']}: F={row['f_score']:.2f}, p={row['p_value']:.2e}")

# Mutual Information
print("\n→ Computing Mutual Information scores...")
selector_mi = SelectKBest(score_func=mutual_info_classif, k=50)
selector_mi.fit(X, y)

mi_scores = pd.DataFrame({
    'feature': X.columns,
    'mi_score': selector_mi.scores_
}).sort_values('mi_score', ascending=False)

mi_features = mi_scores.head(50)['feature'].tolist()

print(f"✓ Selected top 50 features by Mutual Information")
print(f"\nTop 10 features by MI:")
for i, row in mi_scores.head(10).iterrows():
    print(f"  {row['feature']}: MI={row['mi_score']:.4f}")

# ============================================================================
# METHOD 3: Random Forest Feature Importance
# ============================================================================
print("\n" + "="*80)
print("METHOD 3: RANDOM FOREST FEATURE IMPORTANCE")
print("="*80)

print("\nTraining Random Forest for feature importance...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X, y)

rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

rf_features = rf_importance.head(50)['feature'].tolist()

print(f"✓ Selected top 50 features by Random Forest importance")
print(f"\nTop 10 features by importance:")
for i, row in rf_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# METHOD 4: Stability Selection (Bootstrap-based)
# ============================================================================
print("\n" + "="*80)
print("METHOD 4: STABILITY SELECTION")
print("="*80)

print("\nPerforming bootstrap stability selection...")
print("(Running 100 bootstrap iterations...)")

n_bootstrap = 100
n_features_select = 30
stability_counts = np.zeros(X.shape[1])

for i in range(n_bootstrap):
    # Bootstrap sample
    indices = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[indices]
    y_boot = y[indices]
    
    # Fit RFE
    rfe = RFE(
        estimator=LogisticRegression(
            class_weight='balanced',
            max_iter=500,
            solver='liblinear',
            random_state=RANDOM_STATE+i
        ),
        n_features_to_select=n_features_select
    )
    
    rfe.fit(X_boot, y_boot)
    stability_counts += rfe.support_.astype(int)
    
    if (i+1) % 20 == 0:
        print(f"  Progress: {i+1}/{n_bootstrap} iterations")

# Compute stability scores
stability_scores = stability_counts / n_bootstrap

stability_df = pd.DataFrame({
    'feature': X.columns,
    'stability': stability_scores
}).sort_values('stability', ascending=False)

# Select features with stability > 0.5
stable_features = stability_df[stability_df['stability'] > 0.5]['feature'].tolist()

print(f"\n✓ Features with stability > 0.5: {len(stable_features)}")
print(f"\nTop 15 most stable features:")
for i, row in stability_df.head(15).iterrows():
    print(f"  {row['feature']}: {row['stability']:.2f}")

# ============================================================================
# CONSENSUS FEATURE SELECTION
# ============================================================================
print("\n" + "="*80)
print("CONSENSUS FEATURE SELECTION")
print("="*80)

print("\nCombining results from all methods...")

# Count how many times each feature was selected
feature_votes = {}
for feat in X.columns:
    votes = 0
    if feat in rfecv_features:
        votes += 1
    if feat in f_features:
        votes += 1
    if feat in mi_features:
        votes += 1
    if feat in rf_features:
        votes += 1
    if feat in stable_features:
        votes += 1
    feature_votes[feat] = votes

# Sort by votes
consensus_df = pd.DataFrame({
    'feature': list(feature_votes.keys()),
    'votes': list(feature_votes.values())
}).sort_values('votes', ascending=False)

# Select features with >=3 votes
consensus_features = consensus_df[consensus_df['votes'] >= 3]['feature'].tolist()

print(f"\n✓ Features selected by ≥3 methods: {len(consensus_features)}")

print(f"\nConsensus features (voted by ≥3 methods):")
for i, row in consensus_df[consensus_df['votes'] >= 3].head(30).iterrows():
    print(f"  {row['feature']}: {row['votes']}/5 votes")

# ============================================================================
# EVALUATE FEATURE SETS
# ============================================================================
print("\n" + "="*80)
print("EVALUATING FEATURE SETS")
print("="*80)

def evaluate_features(X_subset, y, name):
    """Quick evaluation using cross-validation"""
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear',
        random_state=RANDOM_STATE
    )
    
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    
    auc_scores = cross_val_score(lr, X_subset, y, cv=cv, scoring='roc_auc')
    acc_scores = cross_val_score(lr, X_subset, y, cv=cv, scoring='accuracy')
    
    return {
        'name': name,
        'n_features': X_subset.shape[1],
        'auc_mean': auc_scores.mean(),
        'auc_std': auc_scores.std(),
        'acc_mean': acc_scores.mean(),
        'acc_std': acc_scores.std()
    }

print("\nEvaluating different feature sets...")

results = []

# Original features (from paper: 17 proteins)
original_paper_proteins = ['KRT19', 'HAOX1', 'CXCL1', 'RARRES2', 'FCRL6', 'REN', 'IL13',
                           'SPON1', 'MMP-1', 'ARNT', 'TNFSF13B', 'PRKCQ', 'TRAIL-R2',
                           'hOSCAR', 'MCP-2', 'DPP10', 'GDNF']
original_features = [f for f in original_paper_proteins if f in X.columns] + ['male', 'BLDAS']
if len(original_features) > 2:  # At least some proteins found
    results.append(evaluate_features(X[original_features], y, "Original (Paper)"))

# RFECV features
results.append(evaluate_features(X[rfecv_features], y, "RFECV"))

# Top 50 F-score features
results.append(evaluate_features(X[f_features], y, "F-Score Top 50"))

# Top 50 MI features
results.append(evaluate_features(X[mi_features], y, "MI Top 50"))

# Top 50 RF features  
results.append(evaluate_features(X[rf_features], y, "RF Importance Top 50"))

# Stable features
if len(stable_features) > 10:
    results.append(evaluate_features(X[stable_features], y, "Stable (>0.5)"))

# Consensus features
results.append(evaluate_features(X[consensus_features], y, "Consensus (≥3 votes)"))

# Display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('auc_mean', ascending=False)

print("\n" + "="*80)
print("FEATURE SET COMPARISON")
print("="*80)
print(f"\n{'Feature Set':<25} {'N Features':<12} {'AUC (CV)':<15} {'Accuracy (CV)':<15}")
print("-"*80)
for _, row in results_df.iterrows():
    print(f"{row['name']:<25} {row['n_features']:<12} "
          f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}   "
          f"{row['acc_mean']:.4f}±{row['acc_std']:.4f}")

# Save best feature sets
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save feature selection results
feature_sets = {
    'rfecv': rfecv_features,
    'f_score_top50': f_features,
    'mi_top50': mi_features,
    'rf_top50': rf_features,
    'stable': stable_features,
    'consensus': consensus_features
}

import json
with open('/home/claude/selected_features.json', 'w') as f:
    json.dump(feature_sets, f, indent=2)

# Save consensus scores
consensus_df.to_csv('/home/claude/feature_selection_consensus.csv', index=False)
results_df.to_csv('/home/claude/feature_set_evaluation.csv', index=False)

print("✓ Saved selected features to: selected_features.json")
print("✓ Saved consensus scores to: feature_selection_consensus.csv")
print("✓ Saved evaluation results to: feature_set_evaluation.csv")

# Determine best feature set
best_set = results_df.iloc[0]
print(f"\n🏆 BEST FEATURE SET: {best_set['name']}")
print(f"   Features: {int(best_set['n_features'])}")
print(f"   AUC: {best_set['auc_mean']:.4f} ± {best_set['auc_std']:.4f}")
print(f"   Accuracy: {best_set['acc_mean']:.4f} ± {best_set['acc_std']:.4f}")

print("\n" + "="*80)
print("FEATURE SELECTION COMPLETE")
print("="*80)
