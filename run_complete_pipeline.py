#!/usr/bin/env python3
"""
ATRPred Enhancement - Complete Pipeline Runner
==============================================

Runs the complete analysis pipeline from data loading to final results.

Usage:
    python run_complete_pipeline.py
    python run_complete_pipeline.py --seed 42 --cv-folds 5

Author: [Your Name]
Date: December 2025
"""

import argparse
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run ATRPred Enhancement Pipeline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--cv-folds', type=int, default=5, help='CV folds (default: 5)')
    parser.add_argument('--skip-feature-eng', action='store_true', 
                       help='Skip feature engineering if already done')
    parser.add_argument('--skip-feature-sel', action='store_true',
                       help='Skip feature selection if already done')
    args = parser.parse_args()
    
    print("="*80)
    print("ATRPred Enhancement - Complete Pipeline")
    print("="*80)
    print(f"Random Seed: {args.seed}")
    print(f"CV Folds: {args.cv_folds}")
    print()
    
    start_time = time.time()
    
    # Check data exists
    data_file = Path('data/raw/ra_tot.txt')
    if not data_file.exists():
        print("ERROR: Data file not found!")
        print(f"Expected: {data_file}")
        print("\nPlease download data from: https://github.com/ShuklaLab/ATRPred")
        sys.exit(1)
    
    # Step 1: Feature Engineering
    if not args.skip_feature_eng:
        print("\n[STEP 1/4] Feature Engineering...")
        print("-" * 80)
        import quick_win_implementation as step1
        print("✓ Feature engineering complete")
    else:
        print("\n[STEP 1/4] Skipping feature engineering (--skip-feature-eng)")
    
    # Step 2: Feature Selection
    if not args.skip_feature_sel:
        print("\n[STEP 2/4] Feature Selection...")
        print("-" * 80)
        import feature_selection as step2
        print("✓ Feature selection complete")
    else:
        print("\n[STEP 2/4] Skipping feature selection (--skip-feature-sel)")
    
    # Step 3: Ensemble Models
    print("\n[STEP 3/4] Ensemble Modeling...")
    print("-" * 80)
    import ensemble_models as step3
    print("✓ Ensemble modeling complete")
    
    # Step 4: Final Optimization
    print("\n[STEP 4/4] Final Optimization...")
    print("-" * 80)
    import final_optimization as step4
    print("✓ Final optimization complete")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nResults saved to:")
    print(f"  - results/final_results.json")
    print(f"  - results/model_parameters.json")
    print(f"  - results/final_comparison.csv")
    print(f"  - docs/FINAL_REPORT.md")
    print("\n✓ Ready for publication!")

if __name__ == '__main__':
    main()
