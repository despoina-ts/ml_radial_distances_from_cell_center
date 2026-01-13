# -*- coding: utf-8 -*-
"""
Evaluation utilities for classification models.

This module provides:
- Test-set evaluation (accuracy, macro-F1, confusion matrix)
- Stratified cross-validation
- Saving of metrics in JSON and CSV formats
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

LABELS_FIXED = ["central", "intermediate", "peripheral"]


def evaluate_on_test(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, object]:
    """
    Evaluate predictions on a held-out test set.

    Parameters
    ----------
    y_true : pandas.Series
        Ground-truth labels.
    y_pred : pandas.Series
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, macro-F1, classification report,
        and confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_FIXED)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
    }


def cross_validate_macro_f1(
    pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Perform stratified cross-validation using macro-F1 score.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Preprocessing + classifier pipeline.
    x : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target labels.
    random_state : int
        Random seed for reproducibility.
    n_splits : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    numpy.ndarray
        Macro-F1 scores for each fold.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, x, y, scoring="f1_macro", cv=cv, n_jobs=-1)
    return scores


def save_metrics_json(
    outdir: str,
    model_name: str,
    test_metrics: Dict[str, object],
    cv_scores: np.ndarray,
) -> None:
    """
    Save evaluation metrics in JSON format.

    Parameters
    ----------
    outdir : str
        Output directory.
    model_name : str
        Name of the trained model.
    test_metrics : dict
        Dictionary returned by `evaluate_on_test`.
    cv_scores : numpy.ndarray
        Cross-validation macro-F1 scores.
    """
    payload = {
        "model": model_name,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "labels": LABELS_FIXED,
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_scores": [float(v) for v in cv_scores],
    }

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)


def save_metrics_csvs(
    outdir: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    test_metrics: Dict[str, object],
) -> None:
    """
    Save evaluation metrics (precision, recall, F1, support, confusion matrix) as CSV files.

    Parameters
    ----------
    outdir : str
        Output directory.
    y_true : pandas.Series
        Ground-truth labels.
    y_pred : pandas.Series
        Predicted labels.
    test_metrics : dict
        Dictionary returned by `evaluate_on_test`.
    """
    accuracy = float(test_metrics["accuracy"])
    macro_f1 = float(test_metrics["macro_f1"])
    cm = test_metrics["confusion_matrix"]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS_FIXED, zero_division=0
    )

    rows = [
        {"metric": "accuracy", "class": "", "value": accuracy},
        {"metric": "macro_f1", "class": "", "value": macro_f1},
    ]

    for i, label in enumerate(LABELS_FIXED):
        rows.extend(
            [
                {"metric": "precision", "class": label, "value": float(precision[i])},
                {"metric": "recall", "class": label, "value": float(recall[i])},
                {"metric": "f1", "class": label, "value": float(f1[i])},
                {"metric": "support", "class": label, "value": int(support[i])},
            ]
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(outdir, "model_metrics.csv"), index=False)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in LABELS_FIXED],
        columns=[f"pred_{c}" for c in LABELS_FIXED],
    )
    cm_df.to_csv(os.path.join(outdir, "confusion_matrix.csv"))
