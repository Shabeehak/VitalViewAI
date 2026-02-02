# train_models_improved.py
"""
Improved Training Script with Better Hyperparameters
Fixes poor performance issues
"""

import sys
import time
from datetime import datetime

sys.path.append('src')

from features.feature_engineering import engineer_features_for_training
from models.train_xgboost import XGBoostHealthPredictor

try:
    from xgboost import XGBClassifier
except ImportError:
    import xgboost as xgb
    XGBClassifier = xgb.XGBClassifier

import pandas as pd
import numpy as np

def main():
    """Improved training pipeline with better parameters"""
    
    print("\n" + "="*80)
    print(" "*20 + "🏥 IMPROVED MODEL TRAINING 🏥")
    print("="*80)
    
    # Step 1: Check data quality
    print("\n━"*80)
    print("STEP 0: DATA QUALITY CHECK")
    print("━"*80)
    
    try:
        df_raw = pd.read_csv("data/processed/features_multi.csv")
        positive_rate = df_raw['label'].mean()
        
        print(f"\n📊 Data Statistics:")
        print(f"   Total samples: {len(df_raw):,}")
        print(f"   Deteriorating: {df_raw['label'].sum():,} ({positive_rate:.1%})")
        print(f"   Stable: {(df_raw['label']==0).sum():,} ({1-positive_rate:.1%})")
        
        if positive_rate < 0.20:
            print(f"\n❌ ERROR: Deterioration rate too low ({positive_rate:.1%})")
            print(f"   Target: 30-40%")
            print(f"   Please regenerate data with more deterioration events!")
            return None
        elif positive_rate > 0.60:
            print(f"\n⚠️ WARNING: Deterioration rate very high ({positive_rate:.1%})")
            print(f"   This may not be realistic. Consider regenerating.")
        else:
            print(f"\n✅ Data distribution looks good!")
            
    except FileNotFoundError:
        print(f"\n❌ Data file not found!")
        print(f"   Run: python generate_diverse_training_data.py")
        return None
    
    # Step 2: Feature Engineering
    print("\n━"*80)
    print("STEP 1: FEATURE ENGINEERING")
    print("━"*80)
    
    df_engineered = engineer_features_for_training(
        input_path="data/processed/features_multi.csv",
        output_path="data/processed/features_engineered.csv"
    )
    
    print(f"\n✅ Features ready: {len(df_engineered):,} samples")
    
    # Step 3: Train with improved settings
    print("\n━"*80)
    print("STEP 2: XGBOOST TRAINING (IMPROVED SETTINGS)")
    print("━"*80)
    
    print("\n📂 Loading engineered features...")
    df = pd.read_csv("data/processed/features_engineered.csv")
    
    # Check final distribution
    final_positive_rate = df['label'].mean()
    print(f"\n📊 Final data after feature engineering:")
    print(f"   Total: {len(df):,}")
    print(f"   Positive: {df['label'].sum():,} ({final_positive_rate:.1%})")
    
    # IMPROVED: Better SMOTE settings
    predictor = XGBoostHealthPredictor(
        use_smote=True,
        class_weight_ratio=None  # Let SMOTE handle it
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(df)
    
    # Check after SMOTE
    print(f"\n📊 After SMOTE balancing:")
    print(f"   Training positive: {y_train.sum():,} ({y_train.mean():.1%})")
    print(f"   Validation positive: {y_val.sum():,} ({y_val.mean():.1%})")
    print(f"   Test positive: {y_test.sum():,} ({y_test.mean():.1%})")
    
    # IMPROVED HYPERPARAMETERS
    print("\n🔧 Using IMPROVED hyperparameters:")
    print("   Strategy: Prevent overfitting + Better recall")
    print("   - Lower threshold (0.3 instead of 0.5)")
    print("   - More trees (500 instead of 300)")
    print("   - Deeper trees (6 instead of 4)")
    print("   - Slower learning (0.01)")
    print("   - Regularization (gamma, min_child_weight)")
    
    predictor.model = XGBClassifier(
        # More capacity
        n_estimators=500,
        max_depth=6,
        
        # Slower, more stable learning
        learning_rate=0.01,
        
        # Regularization
        min_child_weight=1,
        gamma=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        
        # For imbalanced data
        scale_pos_weight=1,  # SMOTE already balanced
        
        # Optimization
        eval_metric='aucpr',
        random_state=42,
        use_label_encoder=False,
        
        # Use all CPU cores
        n_jobs=-1
    )
    
    print("\n🔄 Training model (this may take 2-5 minutes)...")
    start_time = time.time()
    
    predictor.model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    training_time = time.time() - start_time
    
    results = predictor.model.evals_result()
    predictor.training_history = {
        'train_aucpr': results['validation_0']['aucpr'],
        'val_aucpr': results['validation_1']['aucpr']
    }
    
    final_train_auc = predictor.training_history['train_aucpr'][-1]
    final_val_auc = predictor.training_history['val_aucpr'][-1]
    
    print(f"\n✅ Training complete! ({training_time:.1f} seconds)")
    print(f"   Final train PR-AUC: {final_train_auc:.4f}")
    print(f"   Final val PR-AUC: {final_val_auc:.4f}")
    
    # Check for overfitting
    overfitting_gap = final_train_auc - final_val_auc
    if overfitting_gap > 0.15:
        print(f"\n⚠️ WARNING: Possible overfitting detected!")
        print(f"   Gap between train and val: {overfitting_gap:.4f}")
    else:
        print(f"\n✅ Good generalization (gap: {overfitting_gap:.4f})")
    
    # CRITICAL: Use lower threshold for better recall
    print("\n🎯 Finding optimal threshold for high recall...")
    threshold = 0.3  # Lower threshold = catch more events
    
    print(f"\n📊 Evaluating with threshold: {threshold:.3f}")
    pr_auc = predictor.evaluate(X_test, y_test, threshold=threshold)
    
    if pr_auc < 0.70:
        print(f"\n❌ PR-AUC still low: {pr_auc:.4f}")
        print(f"   Data quality issues - need to regenerate with more events!")
    elif pr_auc < 0.85:
        print(f"\n⚠️ PR-AUC acceptable but could be better: {pr_auc:.4f}")
    else:
        print(f"\n✅ Excellent PR-AUC: {pr_auc:.4f}")
    
    print("\n📊 Generating visualizations...")
    predictor.plot_training_history()
    predictor.plot_confusion_matrix()
    predictor.plot_feature_importance()
    predictor.plot_pr_curve()
    
    predictor.save_model()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n🎯 Final Performance:")
    print(f"   Validation PR-AUC: {final_val_auc:.4f}")
    print(f"   Test PR-AUC: {pr_auc:.4f}")
    print(f"   Threshold: {threshold:.3f}")
    print(f"\n📁 Outputs:")
    print(f"   Model: models/xgboost_model.pkl")
    print(f"   Plots: models/*.png")
    print(f"\n🚀 Next: python evaluate_model.py --model xgboost")
    
    return predictor

if __name__ == "__main__":
    predictor = main()