"""Exploratory analysis fallback script.

If Prism hides .ipynb files, run this script or copy its cells into a notebook.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import TARGET, label_clinical_groups, read_cardio_data

DATA_PATH = PROJECT_ROOT / 'prism-uploads' / 'cardio_train.csv'
frame = read_cardio_data(DATA_PATH)
print('Dataset shape:', frame.shape)
print('\nTarget balance:')
print(frame[TARGET].value_counts().sort_index())
print('\nData quality:')
print(pd.DataFrame({'dtype': frame.dtypes.astype(str), 'missing': frame.isna().sum(), 'unique': frame.nunique()}))
print('\nClinical group event rates:')
grouped = label_clinical_groups(frame)
for column in ['sex_group', 'age_band', 'cholesterol_group', 'blood_pressure_group']:
    print('\n' + column)
    print(grouped.groupby(column, observed=False)[TARGET].agg(n='size', event_rate='mean').reset_index())
