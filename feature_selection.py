"""
feature_selection.py — Feature Selection for VitalViewAI
=========================================================
Run AFTER training. Analyses which of the 133 features
actually matter and optionally retrains with fewer features.

Usage:
    python feature_selection.py --analyse-only
    python feature_selection.py --retrain

Outputs:
    reports/feature_selection/feature_importance.png
    reports/feature_selection/selected_features.json
    models/xgboost_model_selected.pkl  (--retrain only)
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("reports/feature_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')


# ─────────────────────────────────────────────────────────────
# LOAD MODEL — get importances directly from sklearn API
# ─────────────────────────────────────────────────────────────

def load_model_and_importances():
    """
    Load model and extract feature importances.

    Uses model.feature_importances_ (sklearn API) — this is
    the 'gain' metric normalised to sum=1. Always works
    regardless of how the model was saved.

    Also loads feature names from metadata JSON.
    """
    print("=" * 60)
    print("VitalViewAI — Feature Selection")
    print("=" * 60)

    print("\n📂 Loading trained model...")
    model = joblib.load("models/xgboost_model.pkl")

    # ── Get importances directly ──────────────────────────────
    raw_importances = model.feature_importances_
    print(f"   model.feature_importances_ length : {len(raw_importances)}")
    print(f"   Sum of importances                : {raw_importances.sum():.6f}")
    print(f"   Non-zero importances              : {(raw_importances > 0).sum()}")
    print(f"   Top 3 values                      : "
          f"{sorted(raw_importances, reverse=True)[:3]}")

    if raw_importances.sum() == 0:
        print("\n❌ All importances are zero — model may not be fitted properly.")
        print("   Try rerunning: python train_model_xgboost.py")
        sys.exit(1)

    # ── Load feature names from metadata ─────────────────────
    metadata_path = "models/xgboost_model_metadata.json"
    feature_names = None

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        feature_names = metadata.get("feature_names", [])
        print(f"\n   Feature names from metadata       : {len(feature_names)}")
    except FileNotFoundError:
        print(f"\n   ⚠️  {metadata_path} not found")

    # ── Reconcile lengths ─────────────────────────────────────
    n_imp = len(raw_importances)

    if feature_names and len(feature_names) == n_imp:
        print("   ✅ Feature names match importances length")
    elif feature_names and len(feature_names) != n_imp:
        print(f"   ⚠️  Mismatch: {len(feature_names)} names vs "
              f"{n_imp} importances — using positional names")
        feature_names = [f"feature_{i}" for i in range(n_imp)]
    else:
        print(f"   Using positional feature names (f_0 … f_{n_imp-1})")
        feature_names = [f"feature_{i}" for i in range(n_imp)]

    # ── Build importance dataframe ────────────────────────────
    importance_df = pd.DataFrame({
        'feature':    feature_names[:n_imp],
        'importance': raw_importances
    })
    importance_df['importance_pct'] = (
        importance_df['importance'] / importance_df['importance'].sum() * 100
    ).round(4)
    importance_df = importance_df.sort_values(
        'importance', ascending=False
    ).reset_index(drop=True)

    return model, importance_df, feature_names


# ─────────────────────────────────────────────────────────────
# ANALYSE FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def analyse_importance(importance_df):
    """Print importance summary with cumulative coverage."""
    print("\n" + "─" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("─" * 60)

    total = len(importance_df)
    zero  = (importance_df['importance'] == 0).sum()
    print(f"\n   Total features     : {total}")
    print(f"   Non-zero features  : {total - zero}")
    print(f"   Zero-importance    : {zero}")

    # Top 20
    print(f"\n   {'Rank':<5} {'Feature':<38} {'Gain%':<9} {'Cumulative'}")
    print("   " + "─" * 62)
    cumulative = 0.0
    for i, row in importance_df.head(20).iterrows():
        cumulative += row['importance_pct']
        print(f"   {i+1:<5} {row['feature']:<38} "
              f"{row['importance_pct']:<9.3f} {cumulative:.1f}%")

    # How many features cover key thresholds
    cumsum = importance_df['importance_pct'].cumsum()
    print()
    for pct in [80, 90, 95]:
        n = int(np.searchsorted(cumsum.values, pct)) + 1
        n = min(n, total)
        print(f"   Features to explain {pct}% of gain : {n}")

    # Category breakdown — matches your actual feature naming convention
    categories = {
        'Rolling Stats (mean/std/min/max)': lambda f: any(
            f.endswith(f'_{s}_{w}')
            for s in ['mean', 'std', 'min', 'max']
            for w in ['1h', '6h', '12h', '24h']
        ),
        'Lag & Diff':        lambda f: '_lag_' in f or f.endswith('_diff'),
        'Trend / Slope':     lambda f: '_slope_' in f or '_trend' in f,
        'Interactions':      lambda f: f in [
            'mean_arterial_pressure', 'cv_stress_index',
            'respiratory_efficiency', 'pulse_pressure', 'shock_index'
        ],
        'Temporal':          lambda f: any(x in f for x in [
            'hour', 'day_of_week', 'is_weekend', '_sin', '_cos', 'night'
        ]),
        'Lab Features':      lambda f: any(x in f for x in [
            'glucose', 'creatinine', 'hemoglobin', 'wbc',
            'platelet', 'cholesterol', 'triglyceride', 'bun', 'gfr'
        ]),
        'Base Vitals':       lambda f: f in [
            'heart_rate', 'bp_systolic', 'bp_diastolic',
            'spo2', 'respiratory_rate', 'temperature'
        ],
    }

    print("\n   Importance by category:")
    accounted = set()
    for cat_name, cat_fn in categories.items():
        mask = importance_df['feature'].apply(cat_fn)
        cat_pct   = importance_df.loc[mask, 'importance_pct'].sum()
        cat_count = mask.sum()
        bar = '█' * max(1, int(cat_pct / 2))
        print(f"   {cat_name:<36}: {cat_pct:>5.1f}%  "
              f"({cat_count} features)  {bar}")
        accounted.update(importance_df.loc[mask, 'feature'].tolist())

    uncat = importance_df[~importance_df['feature'].isin(accounted)]
    if len(uncat):
        pct = uncat['importance_pct'].sum()
        print(f"   {'Other':<36}: {pct:>5.1f}%  ({len(uncat)} features)")


# ─────────────────────────────────────────────────────────────
# PLOT IMPORTANCE
# ─────────────────────────────────────────────────────────────

def plot_importance(importance_df):
    """Two-panel chart: bar chart of top 30 + cumulative gain curve."""

    top_n = min(30, len(importance_df))
    top   = importance_df.head(top_n).copy()

    # Colour by category
    def colour(feature):
        if any(feature.endswith(f'_{s}_{w}')
               for s in ['mean','std','min','max']
               for w in ['1h','6h','12h','24h']):    return '#3498db'
        if '_lag_' in feature or feature.endswith('_diff'): return '#e67e22'
        if '_slope_' in feature or '_trend' in feature:     return '#9b59b6'
        if feature in ['mean_arterial_pressure','cv_stress_index',
                       'respiratory_efficiency','pulse_pressure',
                       'shock_index']:                       return '#e74c3c'
        if any(x in feature for x in ['hour','day_of_week',
               'is_weekend','_sin','_cos']):                 return '#1abc9c'
        return '#95a5a6'

    colors = [colour(f) for f in top['feature']]

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Feature Importance Analysis — VitalViewAI',
                 fontsize=14, fontweight='bold', y=1.01)

    # Left: horizontal bar
    y_pos = range(len(top))
    axes[0].barh(y_pos, top['importance_pct'],
                 color=colors, edgecolor='white', height=0.78)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top['feature'], fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('% of Total Gain')
    axes[0].set_title(f'Top {top_n} Features by Gain')
    for i, (_, row) in enumerate(top.iterrows()):
        if row['importance_pct'] > 0:
            axes[0].text(row['importance_pct'] + 0.02, i,
                         f"{row['importance_pct']:.2f}%",
                         va='center', fontsize=7.5)

    # Right: cumulative gain
    cumsum = importance_df['importance_pct'].cumsum().values
    x_vals = range(1, len(cumsum) + 1)
    axes[1].plot(x_vals, cumsum, color='#2c3e50', linewidth=2.5)
    axes[1].fill_between(x_vals, cumsum, alpha=0.12, color='#2c3e50')
    for pct, color in [(80, '#e74c3c'), (90, '#e67e22'), (95, '#f1c40f')]:
        n = int(np.searchsorted(cumsum, pct)) + 1
        n = min(n, len(cumsum))
        axes[1].axhline(y=pct, color=color, linestyle='--', alpha=0.8,
                        label=f'{pct}% gain = {n} features')
        axes[1].axvline(x=n, color=color, linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Number of Features (sorted by importance)')
    axes[1].set_ylabel('Cumulative % of Gain Explained')
    axes[1].set_title('Cumulative Feature Importance')
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(0, min(133, len(importance_df)))
    axes[1].set_ylim(0, 101)

    # Legend for bar chart colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Rolling Stats'),
        Patch(facecolor='#e67e22', label='Lag / Diff'),
        Patch(facecolor='#9b59b6', label='Trend / Slope'),
        Patch(facecolor='#e74c3c', label='Interactions'),
        Patch(facecolor='#1abc9c', label='Temporal'),
        Patch(facecolor='#95a5a6', label='Base Vitals / Other'),
    ]
    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / 'feature_importance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: {out}")


# ─────────────────────────────────────────────────────────────
# SELECT FEATURES
# ─────────────────────────────────────────────────────────────

def select_features(importance_df, strategy='cumulative_90'):
    """
    Select features using one of three strategies:

    cumulative_90  — minimum features explaining 90% of gain (recommended)
    top_n          — fixed top-60 features
    threshold      — features above 0.1% gain each
    """
    print("\n" + "─" * 60)
    print("FEATURE SELECTION")
    print("─" * 60)

    # Always start by removing zero-importance features
    non_zero = importance_df[importance_df['importance'] > 0].copy()
    print(f"\n   Non-zero importance features: {len(non_zero)} / {len(importance_df)}")

    cumsum = non_zero['importance_pct'].cumsum()

    if strategy == 'cumulative_90':
        n = int(np.searchsorted(cumsum.values, 90.0)) + 1
        n = min(n, len(non_zero))
        selected = non_zero.head(n)['feature'].tolist()
        covered  = non_zero.head(n)['importance_pct'].sum()
        print(f"\n   Strategy : Keep features covering 90% of gain")
        print(f"   Selected : {len(selected)} features  "
              f"(covers {covered:.1f}% of gain)")

    elif strategy == 'top_n':
        n = min(60, len(non_zero))
        selected = non_zero.head(n)['feature'].tolist()
        covered  = non_zero.head(n)['importance_pct'].sum()
        print(f"\n   Strategy : Top {n} features")
        print(f"   Selected : {len(selected)} features  "
              f"(covers {covered:.1f}% of gain)")

    elif strategy == 'threshold':
        mask     = non_zero['importance_pct'] >= 0.1
        selected = non_zero[mask]['feature'].tolist()
        covered  = non_zero[mask]['importance_pct'].sum()
        print(f"\n   Strategy : Features with >= 0.1% gain each")
        print(f"   Selected : {len(selected)} features  "
              f"(covers {covered:.1f}% of gain)")

    removed_n   = len(importance_df) - len(selected)
    removed_pct = 100.0 - importance_df[
        importance_df['feature'].isin(selected)
    ]['importance_pct'].sum()

    print(f"   Removed  : {removed_n} features  "
          f"({removed_pct:.1f}% of gain — noise)")

    # Save
    output = {
        'selected_features':     selected,
        'n_selected':            len(selected),
        'n_original':            len(importance_df),
        'selection_strategy':    strategy,
        'cumulative_gain_covered': float(
            importance_df[importance_df['feature'].isin(selected)
            ]['importance_pct'].sum()
        )
    }
    path = OUTPUT_DIR / 'selected_features.json'
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n   ✅ Saved: {path}")

    return selected


# ─────────────────────────────────────────────────────────────
# RETRAIN WITH SELECTED FEATURES
# ─────────────────────────────────────────────────────────────

def retrain_with_selected_features(selected_features):
    """
    Retrain XGBoost using only the selected features.
    Saves as xgboost_model_selected.pkl — original is NOT touched.
    """
    print("\n" + "─" * 60)
    print("RETRAINING WITH SELECTED FEATURES")
    print("─" * 60)

    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (average_precision_score,
                                  roc_auc_score, confusion_matrix)

    print("\n📂 Loading features_engineered.csv...")
    df = pd.read_csv("data/processed/features_engineered.csv")
    print(f"   Loaded {len(df):,} rows")

    available = [f for f in selected_features if f in df.columns]
    missing   = [f for f in selected_features if f not in df.columns]
    if missing:
        print(f"   ⚠️  {len(missing)} selected features not in CSV — skipping them")
    print(f"   Using {len(available)} features")

    if len(available) == 0:
        print("❌ No valid features found. Aborting retrain.")
        return None

    X = df[available].fillna(0).values
    y = df['label'].values

    # Patient-level split — same random_state as training
    patients = df['patient_id'].unique()
    train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
    val_p,  test_p  = train_test_split(temp_p,   test_size=0.667, random_state=42)

    train_mask = df['patient_id'].isin(train_p)
    val_mask   = df['patient_id'].isin(val_p)
    test_mask  = df['patient_id'].isin(test_p)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\n   Train : {len(X_train):,} samples")
    print(f"   Val   : {len(X_val):,} samples")
    print(f"   Test  : {len(X_test):,} samples")

    # SMOTE on training only
    print("\n🔄 Applying SMOTE...")
    minority = int(min(y_train.sum(), (y_train == 0).sum()))
    k = min(5, minority - 1)
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE: {len(X_train):,} samples (50/50 balanced)")

    # Train
    print("\n🤖 Training XGBoost with selected features...")
    start = time.time()

    # Hyperparameters tuned for 68 selected features:
    # - Lower learning_rate (0.01) so model learns slowly and
    #   early stopping doesn't trigger prematurely
    # - More estimators (500) to compensate for slow learning rate
    # - Higher early_stopping_rounds (50) gives more patience
    # - Higher gamma (0.1) and min_child_weight (3) reduce overfitting
    #   more aggressively since we already removed noise features
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=3,
        reg_alpha=0.1,       # L1 regularisation
        reg_lambda=1.0,      # L2 regularisation
        scale_pos_weight=1,
        eval_metric='aucpr',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50   # print every 50 iterations so you can see progress
    )

    elapsed     = time.time() - start
    best_iter   = model.best_iteration
    best_score  = model.best_score

    print(f"\n   Early stopping at iteration : {best_iter}")
    print(f"   Best val score              : {best_score:.4f}")
    print(f"   Training time               : {elapsed:.1f}s")

    # Evaluate at default threshold 0.5
    print("\n📊 Evaluation (threshold = 0.5):")
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc  = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    y_pred  = (y_proba >= 0.5).astype(int)
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"   PR-AUC    : {pr_auc:.4f}")
    print(f"   ROC-AUC   : {roc_auc:.4f}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1        : {f1:.4f}")
    print(f"   TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")

    # Save
    model_path = "models/xgboost_model_selected.pkl"
    joblib.dump(model, model_path)

    metadata = {
        'model_type':     'XGBoost_FeatureSelected',
        'n_features':     len(available),
        'feature_names':  available,
        'best_iteration': int(best_iter),
        'performance': {
            'pr_auc':    float(pr_auc),
            'roc_auc':   float(roc_auc),
            'precision': float(precision),
            'recall':    float(recall),
            'f1':        float(f1)
        }
    }
    meta_path = "models/xgboost_model_selected_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n   ✅ Model saved    : {model_path}")
    print(f"   ✅ Metadata saved : {meta_path}")

    return pr_auc



# ─────────────────────────────────────────────────────────────
# OPTIONAL: HYPERPARAMETER TUNING WITH OPTUNA
# ─────────────────────────────────────────────────────────────

def run_optuna_tuning(selected_features, n_trials=20):
    """
    Automated hyperparameter search using Optuna.

    Optuna uses Bayesian optimisation — each trial learns from
    previous results to focus on promising hyperparameter regions.
    Much smarter than grid search which tries every combination.

    n_trials=20 : ~20-40 minutes
    n_trials=50 : ~60-90 minutes

    Usage: python feature_selection.py --tune
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("\n   Optuna not installed. Run: pip install optuna")
        return None

    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score

    print("\n" + "─" * 60)
    print(f"HYPERPARAMETER TUNING — {n_trials} Optuna trials")
    print("─" * 60)

    print("\n📂 Loading data...")
    df = pd.read_csv("data/processed/features_engineered.csv")
    available = [f for f in selected_features if f in df.columns]

    X = df[available].fillna(0).values
    y = df['label'].values

    patients = df['patient_id'].unique()
    train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
    val_p,  test_p  = train_test_split(temp_p,   test_size=0.667, random_state=42)

    train_mask = df['patient_id'].isin(train_p)
    val_mask   = df['patient_id'].isin(val_p)
    test_mask  = df['patient_id'].isin(test_p)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    # SMOTE once — reuse across all trials
    print("🔄 Applying SMOTE (once for all trials)...")
    minority = int(min(y_train.sum(), (y_train == 0).sum()))
    smote = SMOTE(random_state=42, k_neighbors=min(5, minority - 1))
    X_train_s, y_train_s = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE: {len(X_train_s):,} samples")

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 8),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma':             trial.suggest_float('gamma', 0.0, 0.5),
            'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 2.0),
            'scale_pos_weight':  1,
            'eval_metric':       'aucpr',
            'early_stopping_rounds': 30,
            'random_state':      42,
            'n_jobs':            -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train_s, y_train_s,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        y_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_proba)

    print(f"\n🔍 Running {n_trials} trials (this takes ~{n_trials*2} minutes)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_val_score = study.best_value
    print(f"\n   Best val PR-AUC : {best_val_score:.4f}")
    print(f"   Best params     :")
    for k, v in best.items():
        print(f"     {k:<25}: {v}")

    # Final retrain with best params on test set
    print("\n🤖 Retraining with best params...")
    best['scale_pos_weight']       = 1
    best['eval_metric']            = 'aucpr'
    best['early_stopping_rounds']  = 30
    best['random_state']           = 42
    best['n_jobs']                 = -1

    final_model = XGBClassifier(**best)
    final_model.fit(X_train_s, y_train_s,
                    eval_set=[(X_val, y_val)],
                    verbose=False)

    y_proba   = final_model.predict_proba(X_test)[:, 1]
    test_pr   = average_precision_score(y_test, y_proba)
    print(f"   Test PR-AUC     : {test_pr:.4f}")

    # Save
    joblib.dump(final_model, "models/xgboost_model_tuned.pkl")
    result = {
        'best_params':   best,
        'val_pr_auc':    float(best_val_score),
        'test_pr_auc':   float(test_pr),
        'n_trials':      n_trials,
        'n_features':    len(available),
        'feature_names': available
    }
    with open("models/optuna_best_params.json", 'w') as f:
        json.dump(result, f, indent=2)

    print("\n   ✅ Tuned model saved  : models/xgboost_model_tuned.pkl")
    print("   ✅ Best params saved  : models/optuna_best_params.json")
    return test_pr

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Feature Selection for VitalViewAI'
    )
    parser.add_argument('--analyse-only', action='store_true',
                        help='Analyse importance only — no retraining')
    parser.add_argument('--retrain',      action='store_true',
                        help='Retrain with selected features and better hyperparams')
    parser.add_argument('--tune',         action='store_true',
                        help='Automated hyperparameter tuning with Optuna (~40 min)')
    parser.add_argument('--trials',       type=int, default=20,
                        help='Number of Optuna trials (default: 20)')
    parser.add_argument('--strategy',     default='cumulative_90',
                        choices=['cumulative_90', 'top_n', 'threshold'],
                        help='Feature selection strategy (default: cumulative_90)')
    args = parser.parse_args()

    if not args.analyse_only and not args.retrain and not args.tune:
        print("Usage:")
        print("  python feature_selection.py --analyse-only")
        print("  python feature_selection.py --retrain")
        print("  python feature_selection.py --tune            # Optuna, ~40 min")
        print("  python feature_selection.py --tune --trials 50  # deeper search")
        return

    model, importance_df, feature_names = load_model_and_importances()
    analyse_importance(importance_df)
    plot_importance(importance_df)
    selected = select_features(importance_df, strategy=args.strategy)

    if args.retrain:
        if len(selected) == 0:
            print("\n❌ No features selected — cannot retrain.")
            print("   Check that features_engineered.csv exists and")
            print("   that the model has non-zero importances.")
            return

        print("\n" + "=" * 60)
        print("RETRAINING — original model NOT overwritten")
        print("=" * 60)

        new_pr_auc = retrain_with_selected_features(selected)

        if new_pr_auc is not None:
            try:
                with open("models/xgboost_model_metadata.json") as f:
                    orig_meta = json.load(f)
                orig_pr_auc = orig_meta.get(
                    'performance', {}
                ).get('pr_auc', 0.653)
            except Exception:
                orig_pr_auc = 0.653

            print(f"\n{'='*60}")
            print(f"  Original model PR-AUC  : {orig_pr_auc:.4f}")
            print(f"  Selected model PR-AUC  : {new_pr_auc:.4f}")
            delta = new_pr_auc - orig_pr_auc
            icon  = '✅ improved' if delta > 0 else '⚠️  decreased'
            print(f"  Change                 : {delta:+.4f}  {icon}")

            if new_pr_auc >= orig_pr_auc:
                print("\n  → New model is better or equal.")
                print("    To replace original:")
                print("    copy models\\xgboost_model_selected.pkl "
                      "models\\xgboost_model.pkl")
                print("    copy models\\xgboost_model_selected_metadata.json "
                      "models\\xgboost_model_metadata.json")
            else:
                print("\n  → Original model performs better — keep it.")
                print("    Feature-selected model still saved for comparison.")
            print(f"{'='*60}")

    elif args.tune:
        if len(selected) == 0:
            print("\n❌ No features selected — cannot tune.")
            return
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING with Optuna")
        print("Original model NOT overwritten.")
        print("=" * 60)
        tune_pr = run_optuna_tuning(selected, n_trials=args.trials)
        if tune_pr is not None:
            try:
                with open("models/xgboost_model_metadata.json") as f:
                    orig_meta = json.load(f)
                orig_pr = orig_meta.get('performance', {}).get('pr_auc', 0.653)
            except Exception:
                orig_pr = 0.653
            print(f"\n  Original PR-AUC : {orig_pr:.4f}")
            print(f"  Tuned PR-AUC    : {tune_pr:.4f}")
            delta = tune_pr - orig_pr
            print(f"  Change          : {delta:+.4f}  "
                  f"{'✅ improved' if delta > 0 else '⚠️ decreased'}")

    else:
        print("\n" + "=" * 60)
        print("Analysis complete.")
        print(f"Selected features saved to: "
              f"reports/feature_selection/selected_features.json")
        print("\nNext steps:")
        print("  python feature_selection.py --retrain       # ~10 min")
        print("  python feature_selection.py --tune          # ~40 min, best results")
        print("=" * 60)


if __name__ == "__main__":
    main()