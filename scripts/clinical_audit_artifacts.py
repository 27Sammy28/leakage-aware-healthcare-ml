"""Generate clinical utility and subgroup audit artifacts for the manuscript."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.calibration import lacve_weights, weighted_vote
from src.evaluation import classification_metrics, decision_curve_rows, subgroup_rows
from src.leakage_control import stratified_train_test_split
from src.models import BaggedTrees, GaussianNB, LogisticRegressionGD, ShallowDecisionTree
from src.preprocess import TARGET, StandardPreprocessor, label_clinical_groups, read_cardio_data


MODEL_FACTORIES = {
    "LR": lambda: LogisticRegressionGD(10),
    "Decision Tree": lambda: ShallowDecisionTree(4, 300),
    "Bagged Trees": lambda: BaggedTrees(25, 4, 300),
    "Gaussian NB": lambda: GaussianNB(),
}


def fit_heldout_probabilities():
    frame = read_cardio_data()
    y_values = frame[TARGET].to_numpy(int)
    train_index, test_index = stratified_train_test_split(y_values)
    preprocessor = StandardPreprocessor().fit(frame.iloc[train_index])
    x_train = preprocessor.transform(frame.iloc[train_index])
    x_test = preprocessor.transform(frame.iloc[test_index])
    y_train = y_values[train_index]
    y_test = y_values[test_index]

    probability_by_model = {}
    for name, factory in MODEL_FACTORIES.items():
        probability_by_model[name] = factory().fit(x_train, y_train).predict_proba(x_test)

    metric_by_model = {
        name: classification_metrics(y_test, probabilities)
        for name, probabilities in probability_by_model.items()
    }
    weights = lacve_weights(metric_by_model)
    probability_by_model["LACVE"] = weighted_vote(probability_by_model, weights)
    return frame, test_index, y_test, probability_by_model


def export_prediction_rows(output_dir: Path, test_index, y_test, probability_by_model) -> None:
    rows = []
    for model, probabilities in probability_by_model.items():
        for sample_id, true_value, probability in zip(test_index, y_test, probabilities):
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "model": model,
                    "y_true": int(true_value),
                    "y_pred_proba": float(probability),
                    "y_pred": int(probability >= 0.5),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "heldout_predictions_reconstructed.csv", index=False)


def export_decision_curve(output_dir: Path, y_test, probability_by_model) -> None:
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    net_benefit_frame = decision_curve_rows(y_test, probability_by_model, thresholds)
    net_benefit_frame.to_csv(output_dir / "decision_curve_net_benefit.csv", index=False)
    compact = net_benefit_frame[
        net_benefit_frame.model.isin(["LR", "Decision Tree", "Bagged Trees", "LACVE"])
        & net_benefit_frame.threshold.isin([0.2, 0.3, 0.5])
    ]
    compact.pivot(index="model", columns="threshold", values="net_benefit").reset_index().to_csv(
        output_dir / "decision_curve_net_benefit_compact.csv", index=False
    )
    print(compact.pivot(index="model", columns="threshold", values="net_benefit").round(4).to_string())


def export_subgroups(output_dir: Path, frame, test_index, y_test, probability_by_model) -> None:
    test_frame = label_clinical_groups(frame.iloc[test_index].copy()).reset_index(drop=True)
    rows = []
    group_columns = ["sex_group", "age_band", "cholesterol_group", "blood_pressure_group"]
    for model in ["LR", "Decision Tree", "Bagged Trees", "LACVE"]:
        for column in group_columns:
            rows.extend(subgroup_rows(test_frame, y_test, probability_by_model[model], column, model))
    subgroup_frame = pd.DataFrame(rows)
    subgroup_frame.to_csv(output_dir / "subgroup_calibration_performance.csv", index=False)
    lacve = subgroup_frame[subgroup_frame.model.eq("LACVE")].copy()
    compact_columns = [
        "subgroup_type",
        "subgroup",
        "n",
        "event_rate",
        "accuracy",
        "recall",
        "specificity",
        "roc_auc",
        "brier",
        "calibration_slope",
    ]
    lacve[compact_columns].to_csv(output_dir / "subgroup_lacve_compact.csv", index=False)
    print(lacve[["subgroup_type", "subgroup", "n", "event_rate", "accuracy", "roc_auc", "brier", "calibration_slope"]].round(4).to_string(index=False))


def main() -> None:
    output_dir = Path("clinical_audit_artifacts")
    output_dir.mkdir(exist_ok=True)
    frame, test_index, y_test, probability_by_model = fit_heldout_probabilities()
    export_prediction_rows(output_dir, test_index, y_test, probability_by_model)
    export_decision_curve(output_dir, y_test, probability_by_model)
    export_subgroups(output_dir, frame, test_index, y_test, probability_by_model)
    print("wrote clinical_audit_artifacts")


if __name__ == "__main__":
    main()
