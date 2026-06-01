#!/usr/bin/env python3
"""Run available dependency-light audits and refresh results/."""

from __future__ import annotations

import subprocess
import sys


COMMANDS = [
    [sys.executable, "primary_calibration_ensemble_audit.py"],
    [sys.executable, "stronger_ensemble_baselines.py"],
    [sys.executable, "uci_external_validation.py"],
    [sys.executable, "clinical_audit_artifacts.py"],
    [sys.executable, "scripts/organize_results.py"],
]


def main() -> None:
    for command in COMMANDS:
        print("\n$", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
