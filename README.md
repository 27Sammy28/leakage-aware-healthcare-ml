# LACVE: Leakage-Aware Calibrated Voting Ensemble for Reliable Healthcare Machine Learning

<p align="center">
  <b>Robust • Reproducible • Calibration-Aware • Leakage-Safe Healthcare AI</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg">
  <img src="https://img.shields.io/badge/Focus-Trustworthy%20AI-green.svg">
  <img src="https://img.shields.io/badge/Domain-Healthcare%20ML-red.svg">
  <img src="https://img.shields.io/badge/Status-Research%20Framework-purple.svg">
</p>

---

# Overview

LACVE (Leakage-Aware Calibrated Voting Ensemble) is a reproducible healthcare machine learning framework designed for robust cardiovascular disease prediction under realistic evaluation conditions.

The framework emphasizes:

* leakage-aware evaluation
* calibration-sensitive benchmarking
* reproducibility auditing
* robustness-oriented validation
* trustworthy healthcare AI

rather than accuracy optimization alone.

This repository investigates how methodological issues such as train-test leakage, calibration instability, class imbalance, and dataset shift can significantly distort healthcare machine learning performance.

---

# Research Motivation

Healthcare AI systems are highly sensitive to evaluation mistakes.

Small methodological errors can produce artificially inflated metrics while reducing real-world reliability and clinical trustworthiness.

Common failure modes include:

* preprocessing leakage
* fold contamination
* unstable probability calibration
* threshold sensitivity
* noisy labels
* class imbalance
* domain shift
* overfitting to benchmark datasets

LACVE was developed to explore reliability-aware machine learning evaluation strategies for safer and more reproducible healthcare prediction systems.

---

# Key Features

## Leakage-Aware Evaluation

* fold-safe preprocessing
* train-only normalization
* leakage-safe scaling
* train-only oversampling
* isolated held-out testing

---

## Calibration and Reliability Analysis

* Brier score analysis
* Expected Calibration Error (ECE)
* calibration slope auditing
* reliability diagnostics
* probability-quality evaluation

---

## Robustness-Oriented Validation

* stratified cross-validation
* bootstrap uncertainty estimation
* dataset-shift stress testing
* transportability evaluation
* sensitivity analysis

---

## Reproducible Research Pipeline

* deterministic experimental setup
* fold-level metric export
* saved prediction probabilities
* benchmarking metadata
* reproducible evaluation scripts

---

# Framework Architecture

```text id="ypl7km"
Input Dataset
      │
      ▼
Leakage-Aware Split
      │
      ▼
Fold-Safe Preprocessing
      │
      ▼
Optional Resampling
(SMOTE / ADASYN)
      │
      ▼
Base Learners
(LR, SVM, NB, Tree)
      │
      ▼
Calibration + Validation Audit
      │
      ▼
LACVE Weighted Ensemble
      │
      ▼
Final Predictions + Reliability Metrics
```

---

# Evaluated Models

| Model                | Type                    | Purpose                    |
| -------------------- | ----------------------- | -------------------------- |
| Logistic Regression  | Linear Probabilistic    | Transparent baseline       |
| Linear SVM           | Margin-Based            | Ranking comparator         |
| Gaussian Naive Bayes | Probabilistic           | Lightweight baseline       |
| Nearest Centroid     | Distance-Based          | Prototype classifier       |
| Decision Tree        | Nonlinear Interpretable | Threshold-based modeling   |
| Bagged Trees         | Ensemble                | Variance reduction         |
| LACVE Ensemble       | Weighted Voting         | Reliability-aware ensemble |

---

# Evaluation Metrics

## Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* Balanced Accuracy
* Matthews Correlation Coefficient (MCC)
* Cohen’s Kappa
* Hamming Loss

---

## Ranking Metrics

* ROC-AUC
* PR-AUC

---

## Calibration Metrics

* Brier Score
* Expected Calibration Error (ECE)
* Calibration Slope

---

# Dataset Information

## Primary Cardiovascular Dataset

| Property       | Value                   |
| -------------- | ----------------------- |
| Samples        | 70,000                  |
| Features       | 11 predictive variables |
| Missing Values | 0                       |
| Target         | Cardiovascular disease  |
| Class Balance  | Nearly balanced         |

---

## External Stress-Test Dataset

| Property       | Value                       |
| -------------- | --------------------------- |
| Dataset        | UCI Heart Disease           |
| Samples        | 920                         |
| Source Cohorts | 4                           |
| Validation     | Leave-One-Source-Out        |
| Purpose        | Transportability evaluation |

---

# Main Results

## Held-Out Performance

| Model               | Accuracy | F1     | ROC-AUC | PR-AUC | Brier  |
| ------------------- | -------- | ------ | ------- | ------ | ------ |
| Decision Tree       | 0.7311   | 0.7296 | 0.7823  | 0.7780 | 0.1843 |
| Logistic Regression | 0.7206   | 0.7084 | 0.7843  | 0.7594 | 0.1922 |
| Linear SVM          | 0.6518   | 0.6369 | 0.7064  | 0.6896 | 0.2272 |
| Nearest Centroid    | 0.6422   | 0.6180 | 0.6973  | 0.6743 | 0.2222 |
| Gaussian NB         | 0.5919   | 0.4304 | 0.6875  | 0.6649 | 0.2812 |

---

# Cross-Validation Results

| Model               | Accuracy        | ROC-AUC         | Brier           |
| ------------------- | --------------- | --------------- | --------------- |
| LACVE Ensemble      | 0.7302 ± 0.0023 | 0.7950 ± 0.0036 | 0.1843 ± 0.0015 |
| Bagged Trees        | 0.7278 ± 0.0025 | 0.7944 ± 0.0028 | 0.1842 ± 0.0014 |
| Decision Tree       | 0.7211 ± 0.0019 | 0.7856 ± 0.0025 | 0.1856 ± 0.0011 |
| Logistic Regression | 0.7204 ± 0.0066 | 0.7838 ± 0.0061 | 0.1921 ± 0.0025 |

---

# Key Findings

* leakage-safe evaluation significantly improves benchmarking reliability
* calibration quality is essential for healthcare risk prediction
* simple transparent models remain highly competitive
* ensemble weighting improves robustness modestly but consistently
* dataset shift substantially reduces transportability performance
* probability quality matters beyond threshold accuracy alone

---

# Repository Structure

```bash id="a8d2pc"
LACVE/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   ├── calibration/
│   ├── robustness/
│   ├── visualization/
│   └── utilities/
│
├── experiments/
├── results/
├── figures/
├── configs/
├── scripts/
├── tests/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Installation

```bash id="af0jxt"
git clone https://github.com/yourusername/LACVE.git

cd LACVE

pip install -r requirements.txt
```

---

# Example Usage

```python id="jfp9re"
python train.py

python evaluate.py

python calibration_analysis.py
```

---

# Research Direction

This work aligns with broader research interests in:

* trustworthy AI
* uncertainty-aware machine learning
* robust healthcare prediction
* reproducible AI systems
* calibration-aware healthcare AI
* reliability-oriented benchmarking
* interpretable biomedical machine learning

---

# Future Work

Planned extensions include:

* advanced calibration analysis
* Bayesian uncertainty estimation
* external clinical validation
* domain adaptation
* uncertainty quantification
* explainable AI integration
* synthetic noise robustness testing
* distribution-shift benchmarking

---

# Disclaimer

This repository is intended strictly for research and benchmarking purposes.

The models and outputs provided here are not approved for clinical diagnosis or medical decision-making.

---

# Citation

```bibtex id="ewg7xl"
@article{worku2026lacve,
  title={LACVE: Leakage-Aware Calibrated Voting Ensemble for Reliable Healthcare Machine Learning},
  author={Worku, Samuel},
  year={2026}
}
```

---

# Author

Samuel Worku

Research Interests:

* trustworthy AI
* healthcare machine learning
* uncertainty-aware prediction
* calibration-aware systems
* reproducible AI evaluation
* robustness-oriented machine learning
