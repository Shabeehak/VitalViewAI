#verify_dataset.py
import pandas as pd

print('='*70)
print('FINAL DATASET VERIFICATION')
print('='*70)

features_df = pd.read_csv('data/processed/features_multi.csv')
lab_df = pd.read_csv('data/processed/lab_results_multi.csv')

# Class distribution
pos = (features_df['label'] == 1).sum()
neg = (features_df['label'] == 0).sum()
pct = pos / len(features_df) * 100

print(f'\n📊 Dataset Summary:')
print(f'   Total samples: {len(features_df):,}')
print(f'   Features: {len(features_df.columns) - 2}')
print(f"   Patients: {features_df['patient_id'].nunique()}")

print(f'\n📈 Class Distribution:')
print(f'   Deterioration: {pos:,} ({pct:.2f}%)')
print(f'   Stable: {neg:,} ({100-pct:.2f}%)')
print(f'   Imbalance ratio: {neg/pos:.1f}:1')

# Profile stats
print(f'\n👥 Patient Profile Stats:')
for profile in ['healthy', 'at_risk', 'deteriorating']:
    patients = [p for p in features_df['patient_id'].unique() if profile in p]
    if patients:
        count = len(patients)
        profile_data = features_df[features_df['patient_id'].isin(patients)]
        avg_hr = profile_data['heart_rate'].mean()
        avg_bp = profile_data['bp_systolic'].mean()
        pos_rate = (profile_data['label'] == 1).sum() / len(profile_data) * 100
        print(f'   {profile.capitalize():15s}: {count:2d} patients | HR={avg_hr:5.1f} | BP={avg_bp:5.1f} | {pos_rate:4.1f}% deterioration')

# Lab variation check
print(f'\n🧪 Lab Data Quality:')
print(f'   Total lab tests: {len(lab_df)}')
print(f"   Unique patients: {lab_df['patient_id'].nunique()}")
print(f"   Glucose variation (std): {lab_df['glucose_fasting'].std():.2f}")
print(f'   Sample lab values:')
for i in range(min(5, len(lab_df))):
    pid = lab_df.iloc[i]['patient_id']
    glucose = lab_df.iloc[i]['glucose_fasting']
    print(f'      {pid}: {glucose:.2f} mg/dL')

print(f'\n✅ Quality Checks:')
checks = [
    ('Class balance', 1.0 <= pct <= 10.0, f'{pct:.1f}% (target: 1-10%)'),
    ('Enough samples', len(features_df) > 100000, f'{len(features_df):,} samples'),
    ('All patients', features_df['patient_id'].nunique() == 50, f"{features_df['patient_id'].nunique()} patients"),
    ('Lab diversity', lab_df['glucose_fasting'].std() > 3.0, f"std={lab_df['glucose_fasting'].std():.1f}"),
    ('Profile mix', len([p for p in features_df['patient_id'].unique() if 'healthy' in p]) == 30, 'Profile distribution')
]

for check_name, passed, detail in checks:
    status = '✅' if passed else '❌'
    print(f'   {status} {check_name:20s}: {detail}')

if all(c[1] for c in checks):
    print(f'\n🎉 ALL CHECKS PASSED!')
    print(f'\n🚀 Ready for model training!')
    print(f'   Run: python train_models.py')
else:
    print(f'\n⚠️  Some checks failed')

print('='*70)