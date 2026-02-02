# train_models_quick.py
"""
Quick Training Script - XGBoost Only
Fixes overfitting and skips memory-intensive LSTM
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

def main():
    """Quick training pipeline - XGBoost only"""
    
    print("\n" + "="*80)
    print(" "*20 + "🏥 QUICK MODEL TRAINING (XGBoost Only) 🏥")
    print("="*80)
    
    # Step 1: Feature Engineering
    print("\n━"*80)
    print("STEP 1/2: FEATURE ENGINEERING")
    print("━"*80)
    
    df_engineered = engineer_features_for_training(
        input_path="data/processed/features_multi.csv",
        output_path="data/processed/features_engineered.csv"
    )
    
    print(f"\n✅ Features ready: {len(df_engineered)} samples")
    
    # Step 2: Train XGBoost with FIXED hyperparameters
    print("\n━"*80)
    print("STEP 2/2: XGBOOST TRAINING (OVERFITTING FIXES)")
    print("━"*80)
    
    print("\n📂 Loading engineered features...")
    df = pd.read_csv("data/processed/features_engineered.csv")
    
    # FIXED: Stronger regularization to prevent overfitting
    predictor = XGBoostHealthPredictor(use_smote=True, class_weight_ratio=10)
    
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(df)
    
    # FIXED HYPERPARAMETERS: Prevent overfitting
    print("\n🔧 Using overfitting prevention settings:")
    print("   - Lower learning rate (0.01 instead of 0.1)")
    print("   - More estimators (300 instead of 100)")
    print("   - Lower max_depth (4 instead of 6)")
    print("   - Added min_child_weight and gamma")
    
    # Update model parameters to prevent overfitting
    predictor.model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.01,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # SMOTE already balanced classes
        eval_metric='aucpr',
        random_state=42,
        use_label_encoder=False
    )
    
    print("\n🔄 Training model...")
    predictor.model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    results = predictor.model.evals_result()
    predictor.training_history = {
        'train_aucpr': results['validation_0']['aucpr'],
        'val_aucpr': results['validation_1']['aucpr']
    }
    
    print(f"\n✅ Training complete!")
    print(f"   Final train PR-AUC: {predictor.training_history['train_aucpr'][-1]:.4f}")
    print(f"   Final val PR-AUC: {predictor.training_history['val_aucpr'][-1]:.4f}")
    
    pr_auc = predictor.evaluate(X_test, y_test, threshold=0.3)  # Lower threshold for imbalanced data
    
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
    print(f"   PR-AUC: {pr_auc:.4f}")
    print(f"\n📁 Outputs:")
    print(f"   Model: models/xgboost_model.pkl")
    print(f"   Plots: models/*.png")
    print(f"\n🚀 Next: python predictor.py")
    
    return predictor

if __name__ == "__main__":
    predictor = main()