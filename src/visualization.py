"""Plotting helpers for generated audit artifacts."""
from pathlib import Path
import matplotlib.pyplot as plt

def plot_decision_curve(net_benefit_frame, output_path):
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, group in net_benefit_frame.groupby("model"):
        ax.plot(group["threshold"], group["net_benefit"], marker="o", label=model)
    reference = net_benefit_frame.drop_duplicates("threshold")
    ax.plot(reference["threshold"], reference["treat_all_net_benefit"], linestyle="--", color="gray", label="Treat all")
    ax.axhline(0, color="black", linewidth=0.8, label="Treat none")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output_path, dpi=300); plt.close(fig)

def plot_subgroup_auc(subgroup_frame, output_path, model="LACVE"):
    output_path = Path(output_path)
    data = subgroup_frame[subgroup_frame["model"].eq(model)].copy()
    data["label"] = data["subgroup_type"] + ": " + data["subgroup"]
    data = data.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(data))))
    ax.barh(data["label"], data["roc_auc"], color="#2F6B9A")
    ax.set_xlim(0.5, 0.9); ax.set_xlabel("ROC-AUC"); ax.set_title(f"{model} subgroup discrimination")
    fig.tight_layout(); fig.savefig(output_path, dpi=300); plt.close(fig)
