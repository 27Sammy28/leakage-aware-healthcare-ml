"""Data loading and preprocessing helpers for the cardiovascular benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_FEATURES = ["age_years", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
TARGET = "cardio"

class StandardPreprocessor:
    """Train-only standardization for numeric tabular features."""
    def __init__(self, features=None):
        self.features = features or DEFAULT_FEATURES
    def fit(self, frame):
        values = frame[self.features].to_numpy(float)
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self
    def transform(self, frame):
        values = frame[self.features].to_numpy(float)
        return (values - self.mean_) / self.std_
    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)

Prep = StandardPreprocessor

def read_cardio_data(path="prism-uploads/cardio_train.csv"):
    path = Path(path)
    first_line = path.read_text().splitlines()[0]
    separator = ";" if first_line.count(";") else ","
    frame = pd.read_csv(path, sep=separator)
    if "age" in frame.columns and "age_years" not in frame.columns:
        frame["age_years"] = frame["age"] / 365.25
    return frame

def label_clinical_groups(frame):
    output = frame.copy()
    output["sex_group"] = np.where(output["gender"].astype(int).eq(1), "Female/code 1", "Male/code 2")
    output["age_band"] = pd.cut(output["age_years"], bins=[0, 45, 55, 200], labels=["<45", "45--54", "55+"], right=False)
    output["cholesterol_group"] = output["cholesterol"].map({1: "Normal", 2: "Above normal", 3: "Well above normal"}).fillna("Unknown")
    systolic = output["ap_hi"]
    diastolic = output["ap_lo"]
    output["blood_pressure_group"] = np.select(
        [(systolic < 120) & (diastolic < 80), (systolic < 140) & (diastolic < 90), (systolic >= 140) | (diastolic >= 90)],
        ["Lower range", "Elevated/stage 1", "Stage 2/high"],
        default="Implausible/other",
    )
    return output
