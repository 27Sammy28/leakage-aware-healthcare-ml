# LACVE: Leakage-Aware Calibrated Voting Ensemble for Reliable Healthcare Machine Learning

<p align="center">
  <b>Robust • Reproducible • Calibration-Aware • Leakage-Safe Healthcare AI</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Stack-NumPy%20%7C%20SciPy-orange.svg" alt="NumPy and SciPy stack">
  <img src="https://img.shields.io/badge/Focus-Trustworthy%20AI-green.svg" alt="Trustworthy AI">
  <img src="https://img.shields.io/badge/Domain-Healthcare%20ML-red.svg" alt="Healthcare machine learning">
  <img src="https://img.shields.io/badge/Status-Research%20Framework-purple.svg" alt="Research framework">
</p>


---

## Overview

**LACVE** is a leakage-aware calibrated voting ensemble and reproducibility-audit framework for cardiovascular risk prediction research. The project studies how healthcare machine learning benchmarks change when preprocessing, resampling, model weighting, calibration, threshold interpretation, and external validation are handled under explicit leakage-safe rules.

The goal is not to maximize one headline accuracy number. The goal is to make model comparison more reliable, auditable, and clinically cautious by reporting discrimination, fixed-threshold performance, calibration-oriented error, uncertainty, subgroup behavior, and transportability stress tests together.

---

## Research Motivation

Healthcare AI systems are sensitive to evaluation mistakes. Small workflow errors can produce inflated metrics while weakening real-world reliability and clinical trust.

This repository focuses on common healthcare ML failure modes:

- preprocessing leakage from fitting scalers before splitting data
- fold contamination from resampling before cross-validation
- unstable probability calibration and misleading risk estimates
- threshold sensitivity between false negatives and false positives
- benchmark overfitting and weak external transportability
- metric cherry-picking when only the most favorable score is reported
- reproducibility gaps caused by missing splits, predictions, and metadata

LACVE was developed as a reliability-aware evaluation strategy for safer and more reproducible healthcare prediction research.

---

## Core Contribution

LACVE assigns soft-voting weights using validation discrimination, probability error, and calibration-slope deviation:

```text
reliability_m = ROC_AUC_m / (Brier_m + |calibration_slope_m - 1| + epsilon)
```

A base learner receives higher weight only when it ranks cases well, has lower probability error, and produces probabilities with a calibration slope closer to the ideal value of 1. This makes the ensemble calibration-conscious instead of relying on uniform voting or accuracy-only weighting.

---

## Key Features

### Leakage-Aware Evaluation

- leakage-safe train/test splitting
- train-only normalization and scaling
- fold-contained cross-validation
- train-only SMOTE/ADASYN sensitivity analysis
- held-out test isolation for final evaluation

### Calibration and Reliability Analysis

- Brier score analysis
- calibration slope and intercept audit
- probability-quality evaluation
- reliability-focused ensemble weighting
- calibration-aware interpretation of model outputs

### Robustness-Oriented Validation

- stratified cross-validation summaries
- bootstrap uncertainty intervals
- fixed-threshold net-benefit summary
- subgroup audit by clinical covariates
- UCI Heart Disease leave-one-source-out stress testing

### Reproducible Research Pipeline

- deterministic experimental scripts
- fold-level metric exports
- saved benchmark tables and artifact checklist
- organized `results/` directory for GitHub review
- manuscript-ready figures and reports

---

## Evaluated Models

| Model | Type | Role in Benchmark |
| --- | --- | --- |
| Logistic Regression | Linear probabilistic | Transparent calibrated baseline |
| Linear SVM | Margin-based | Ranking and classification comparator |
| Gaussian Naive Bayes | Probabilistic | Lightweight baseline |
| Nearest Centroid | Distance-based | Prototype classifier |
| Shallow Decision Tree | Nonlinear interpretable | Threshold-oriented model |
| Bagged Shallow Trees | Ensemble | Variance-reduction comparator |
| Random-Subspace Trees | Ensemble | Feature-subsampling comparator |
| Extra-Randomized Trees | Ensemble | Stronger package-free tree baseline |
| LACVE | Weighted soft voting | Reliability-aware calibrated ensemble |

---

## Evaluation Metrics

| Metric Family | Metrics |
| --- | --- |
| Classification | Accuracy, precision, recall, F1-score, specificity, balanced accuracy, MCC |
| Ranking | ROC-AUC, PR-AUC |
| Calibration | Brier score, calibration slope, calibration intercept |
| Robustness | Cross-validation mean ± SD, bootstrap intervals, subgroup summaries |
| Utility | Confusion matrices and fixed-threshold net benefit |
| Transportability | UCI cross-validation and leave-one-source-out validation |

---

## Dataset Information

### Primary Cardiovascular Dataset

| Property | Value |
| --- | --- |
| Samples | 70,000 |
| Predictive variables | 11 |
| Missing values | 0 |
| Target | Cardiovascular disease indicator |
| Class balance | Nearly balanced |
| Primary split | 80% train / 20% held-out test |

### External Stress-Test Dataset

| Property | Value |
| --- | --- |
| Dataset | UCI Heart Disease |
| Samples | 920 |
| Source cohorts | Cleveland, Hungary, Switzerland, VA Long Beach |
| Validation | Cross-validation and leave-one-source-out testing |
| Purpose | Transportability stress test |

---

## Main Results

### Held-Out Performance

| Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7311 | 0.7296 | 0.7823 | 0.7780 | 0.1843 |
| Logistic Regression | 0.7206 | 0.7084 | 0.7843 | 0.7594 | 0.1922 |
| Linear SVM | 0.6518 | 0.6369 | 0.7064 | 0.6896 | 0.2272 |
| Nearest Centroid | 0.6422 | 0.6180 | 0.6973 | 0.6743 | 0.2222 |
| Gaussian NB | 0.5919 | 0.4304 | 0.6875 | 0.6649 | 0.2812 |

### Cross-Validation Results

| Model | Accuracy | ROC-AUC | Brier |
| --- | ---: | ---: | ---: |
| LACVE Ensemble | 0.7302 ± 0.0023 | 0.7950 ± 0.0036 | 0.1843 ± 0.0015 |
| Bagged Trees | 0.7278 ± 0.0025 | 0.7944 ± 0.0028 | 0.1842 ± 0.0014 |
| Decision Tree | 0.7211 ± 0.0019 | 0.7856 ± 0.0025 | 0.1856 ± 0.0011 |
| Logistic Regression | 0.7204 ± 0.0066 | 0.7838 ± 0.0061 | 0.1921 ± 0.0025 |

### Stronger Ensemble Audit

The added package-free stronger ensemble audit found that extra-randomized trees achieved the strongest cross-validated accuracy and ROC-AUC among the dependency-light models. This is reported deliberately: the framework is designed to identify when stronger comparators outperform LACVE rather than force a single preferred model narrative.

---

## Key Findings

- Leakage-safe evaluation improves benchmarking reliability and reduces avoidable optimism.
- Calibration quality is essential when model outputs are interpreted as risk scores.
- Transparent and lightly ensembled models remain competitive on structured cardiovascular data.
- LACVE improves robustness modestly while preserving interpretability and auditability.
- Source-level dataset shift reduces transportability, supporting external validation requirements.
- Probability quality, threshold consequences, and clinical utility should be interpreted separately from accuracy.

---

## Repository Structure

```text
.
├── main.tex                         # Manuscript source
├── main.pdf                         # Compiled manuscript
├── src/                             # Reusable model/evaluation utilities
├── scripts/                         # Runnable audit and organization scripts
├── figures/                         # Generated and README figures
├── results/                         # Organized figures, metrics, and reports
├── primary_audit_artifacts/          # Calibration/LACVE audit outputs
├── stronger_baseline_artifacts/      # Stronger ensemble audit outputs
├── uci_external_artifacts/           # UCI stress-test outputs
├── clinical_audit_artifacts/         # Clinical utility/subgroup audit outputs
├── benchmark_results.csv             # Held-out benchmark results
├── fold_metrics.csv                  # Cross-validation fold metrics
└── artifact_checklist.md             # Reproducibility artifact checklist
```

---

## How to Reproduce

Run the available audit helpers from the repository root:

```bash
python scripts/run_all_audits.py
python scripts/organize_results.py
```

Useful focused commands:

```bash
python scripts/run_clinical_audit.py
python scripts/organize_results.py
```

To rebuild the manuscript PDF when a LaTeX toolchain is available:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

---

## Important Outputs

| Output | Description |
| --- | --- |
| `results/figures/` | PNG figures for review and manuscript support |
| `results/metrics/` | CSV metric tables and validation outputs |
| `results/reports/` | Compiled manuscript PDF and audit notes |
| `benchmark_results.csv` | Primary held-out model comparison |
| `fold_metrics.csv` | Cross-validation metrics |
| `artifact_checklist.md` | Reproducibility checklist |

---

## Research Direction

This work supports broader research in:

- trustworthy healthcare AI
- calibration-aware machine learning
- uncertainty-aware prediction
- robust biomedical benchmarking
- interpretable clinical risk modeling
- leakage-safe evaluation protocols
- reproducible AI systems

---

## Future Work

Planned extensions include:

- optimized scikit-learn, XGBoost, LightGBM, and CatBoost comparators
- LACVE ablation studies for AUC-only, Brier-only, and slope-penalty variants
- calibration curves with confidence intervals and ECE by bin
- paired statistical tests such as DeLong and McNemar tests
- external clinical validation on prospectively collected data
- decision-curve analysis across clinically meaningful thresholds
- explanation-stability analysis across folds and model refits
- robustness tests with noisy features, corrupted labels, and alternative splits

---

## Clinical Disclaimer

This repository is intended strictly for research, benchmarking, and reproducibility auditing. The models and outputs are not approved for clinical diagnosis, treatment selection, or medical decision-making. Any clinical use would require prospective validation, clinician-reviewed preprocessing rules, calibrated deployment thresholds, subgroup review, and post-deployment monitoring.

---

## Citation

```bibtex
@article{worku2026lacve,
  title={LACVE: A Leakage-Aware Calibrated Voting Ensemble and Reproducibility-Audit Framework for Digital Cardiovascular Risk Prediction},
  author={Worku, Samuel},
  year={2026},
  note={Research manuscript and reproducibility framework}
}
```

---

## Author

**Samuel Worku**

Research interests: trustworthy AI, healthcare machine learning, uncertainty-aware prediction, calibration-aware systems, reproducible AI evaluation, and robustness-oriented machine learning.
