"""
QUICK WIN TRIO IMPLEMENTATION
===============================
Strategy 1: Advanced Ensemble Learning
Strategy 2: Advanced Feature Engineering  
Strategy 6: Advanced Feature Selection

Target: >90% accuracy, >0.93 AUC, >97% specificity
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("QUICK WIN TRIO: ADVANCED ANTI-TNF RESPONSE PREDICTION")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[STEP 1] Loading data...")

df = pd.read_csv('/mnt/user-data/uploads/ra_tot.txt', sep=' ', quotechar='"')

# Identify columns
metadata_cols = ['StudyID', 'male', 'age', 'dd', 'BLDAS', 'delDAS', 'nice', 
                 'Plate.ID', 'Plate.ID.1', 'Plate.ID.2', 'Plate.ID.3',
                 'QC.Warning', 'QC.Warning.1', 'QC.Warning.2', 'QC.Warning.3']

protein_cols = [col for col in df.columns if col not in metadata_cols]

# Clean data
df_clean = df.dropna(subset=['delDAS']).copy()
df_clean['response'] = (df_clean['delDAS'] < -1.2).astype(int)

print(f"✓ Total samples: {len(df_clean)}")
print(f"✓ Responders: {df_clean['response'].sum()}")
print(f"✓ Non-responders: {len(df_clean) - df_clean['response'].sum()}")
print(f"✓ Total proteins: {len(protein_cols)}")

# ============================================================================
# STEP 2: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 2] Advanced Feature Engineering...")

# 2A: Standardize protein names for consistency
protein_df = df_clean[protein_cols].copy()
protein_df.columns = [col.replace('.', '-') for col in protein_df.columns]
protein_cols_clean = list(protein_df.columns)

# 2B: Create biological pathway features based on paper's findings
print("  → Creating pathway-based features...")

# IL-17 Pathway (from paper's enrichment analysis)
il17_proteins = []
for p in ['IL13', 'CXCL1', 'MMP-1', 'TNFSF13B']:
    if p in protein_cols_clean:
        il17_proteins.append(p)

if len(il17_proteins) > 0:
    protein_df['IL17_pathway_score'] = protein_df[il17_proteins].mean(axis=1)
    print(f"    ✓ IL-17 pathway score ({len(il17_proteins)} proteins)")

# Inflammatory Response Score (high-interaction proteins from paper)
inflammatory_proteins = []
for p in ['CXCL1', 'CCL8', 'IL13', 'MMP-1']:
    if p in protein_cols_clean:
        inflammatory_proteins.append(p)

if len(inflammatory_proteins) > 0:
    protein_df['inflammatory_score'] = protein_df[inflammatory_proteins].sum(axis=1)
    print(f"    ✓ Inflammatory score ({len(inflammatory_proteins)} proteins)")

# Oxidative Stress Score (from your article)
oxidative_proteins = []
for p in ['HAOX1', 'ARNT', 'GT']:
    if p in protein_cols_clean:
        oxidative_proteins.append(p)

if len(oxidative_proteins) > 0:
    protein_df['oxidative_stress_score'] = protein_df[oxidative_proteins].mean(axis=1)
    print(f"    ✓ Oxidative stress score ({len(oxidative_proteins)} proteins)")

# 2C: Protein-Protein Interaction Features (from network analysis)
print("  → Creating PPI-based features...")

# High-interaction hub proteins from paper (Fig 4C)
hub_proteins = {'CXCL1': 0.421, 'IL13': -0.651, 'CCL8': -0.243, 'MMP-1': -0.830}
hub_features = []

for protein, weight in hub_proteins.items():
    if protein in protein_cols_clean:
        hub_features.append(protein)

if len(hub_features) >= 2:
    # Create weighted hub interaction score
    hub_score = np.zeros(len(protein_df))
    for protein, weight in hub_proteins.items():
        if protein in protein_cols_clean:
            hub_score += protein_df[protein].values * weight
    protein_df['hub_interaction_score'] = hub_score
    print(f"    ✓ Hub interaction score ({len(hub_features)} proteins)")

# 2D: Clinical variable interactions
print("  → Creating clinical interaction features...")

X_clinical = df_clean[['male', 'BLDAS']].copy()

# Gender-protein interactions (paper showed gender significance)
if 'IL13' in protein_cols_clean:
    X_clinical['gender_IL13_interaction'] = X_clinical['male'] * protein_df['IL13']
    
if 'CXCL1' in protein_cols_clean:
    X_clinical['gender_CXCL1_interaction'] = X_clinical['male'] * protein_df['CXCL1']

# Gender-BLDAS interaction
X_clinical['gender_BLDAS_interaction'] = X_clinical['male'] * X_clinical['BLDAS']

# BLDAS to protein ratios (clinical relevance)
if 'CXCL1' in protein_cols_clean:
    X_clinical['BLDAS_CXCL1_ratio'] = X_clinical['BLDAS'] / (protein_df['CXCL1'] + 1e-6)

print(f"    ✓ Created {X_clinical.shape[1]} clinical features")

# 2E: Top protein polynomial features (non-linear relationships)
print("  → Creating polynomial features for top proteins...")

# Top proteins from original paper
top_protein_candidates = ['KRT19', 'HAOX1', 'CXCL1', 'FCRL6', 'IL13', 
                          'RARRES2', 'REN', 'ARNT', 'MMP-1']

top_proteins = [p for p in top_protein_candidates if p in protein_cols_clean]

if len(top_proteins) >= 3:
    # Create squared terms for top proteins
    for protein in top_proteins[:5]:  # Limit to top 5 to avoid too many features
        protein_df[f'{protein}_squared'] = protein_df[protein] ** 2
    print(f"    ✓ Created squared terms for {min(5, len(top_proteins))} top proteins")

# 2F: Statistical features
print("  → Creating statistical aggregate features...")

# Mean, std, max of protein groups
protein_stats = pd.DataFrame()
protein_stats['protein_mean'] = protein_df[protein_cols_clean].mean(axis=1)
protein_stats['protein_std'] = protein_df[protein_cols_clean].std(axis=1)
protein_stats['protein_max'] = protein_df[protein_cols_clean].max(axis=1)
protein_stats['protein_min'] = protein_df[protein_cols_clean].min(axis=1)

print(f"    ✓ Created {len(protein_stats.columns)} statistical features")

# Combine all features
print("\n  → Combining all engineered features...")
X_all = pd.concat([protein_df, X_clinical, protein_stats], axis=1)

print(f"\n✓ Total engineered features: {X_all.shape[1]}")
print(f"  - Original proteins: {len(protein_cols_clean)}")
print(f"  - Pathway scores: {3 if len(il17_proteins) > 0 else 0}")
print(f"  - PPI features: {1 if len(hub_features) >= 2 else 0}")  
print(f"  - Clinical interactions: {X_clinical.shape[1] - 2}")
print(f"  - Polynomial features: {min(5, len(top_proteins)) if len(top_proteins) >= 3 else 0}")
print(f"  - Statistical features: {len(protein_stats.columns)}")

# ============================================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================================
print("\n[STEP 3] Handling missing values...")

# Check missing values
missing_before = X_all.isnull().sum().sum()
print(f"  → Missing values before imputation: {missing_before}")

if missing_before > 0:
    # KNN imputation (as per original paper)
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    X_imputed = imputer.fit_transform(X_all)
    X_all = pd.DataFrame(X_imputed, columns=X_all.columns, index=X_all.index)
    
    missing_after = X_all.isnull().sum().sum()
    print(f"  → Missing values after imputation: {missing_after}")
    print(f"  ✓ Imputation complete using k-NN (k=5)")
else:
    print(f"  ✓ No missing values detected")

# ============================================================================
# STEP 4: STANDARDIZATION
# ============================================================================
print("\n[STEP 4] Standardizing features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
X_scaled = pd.DataFrame(X_scaled, columns=X_all.columns, index=X_all.index)

print(f"  ✓ All features standardized (mean=0, std=1)")

# Target variable
y = df_clean['response'].values

print(f"\n✓ Final dataset shape: {X_scaled.shape}")
print(f"✓ Target distribution: {np.bincount(y)}")

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)

# Save preprocessed data for next steps
print("\nSaving preprocessed data...")
X_scaled.to_csv('/home/claude/X_scaled.csv', index=False)
np.save('/home/claude/y.npy', y)
print("✓ Data saved successfully")

