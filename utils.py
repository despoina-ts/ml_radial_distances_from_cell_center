# -*- coding: utf-8 -*-
"""
Utility functions used across the ML pipeline.

This module includes:
- Column-name helpers
- Default column dropping rules
- Simple argument parsing helpers
- A training-set upsampling helper
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def is_start_end(col: str) -> bool:
    """
    Determine whether a column appears to represent a start/end coordinate.

    The check is based on common naming patterns for gene/transcript coordinates
    such as "gene_start", "transcript_end", etc.

    Parameters
    ----------
    col : str
        Column name to evaluate.

    Returns
    -------
    bool
        True if the column name appears to refer to a gene/transcript start or
        end coordinate, otherwise False.
    """
    normalized = col.lower().replace(" ", "").replace("-", "_")
    return bool(re.search(r"(gene|tx|transcript).*(start|end)", normalized))


def default_drop_cols(cols: Sequence[str]) -> List[str]:
    """
    Compute a default list of columns to drop before model training.

    This function removes:
    - Coordinate-like columns that match `is_start_end`
    - Common identifiers and helper columns that should not be used as features
      (e.g., gene IDs, bead coordinates, scope annotations)

    Parameters
    ----------
    cols : sequence of str
        Column names from the input dataframe.

    Returns
    -------
    list of str
        Column names that should be excluded from the feature matrix.
    """
    drop = set()

    for col in cols:
        col_lower = col.lower()

        if is_start_end(col):
            drop.add(col)

        if col_lower in {
            "gene.stable.id",
            "gene.name",
            "bead_id",
            "bead_start",
            "bead_end",
            "scope",
            "scope_hard",
            "scope_balanced",
            "scope_partial",
            "assign_note",
            "overlap_bp",
            "dist_zero_norm",
        }:
            drop.add(col)

    return list(drop)


def parse_optional_int(value: object) -> Optional[int]:
    """
    Parse a value that may represent an integer or the string "None".

    Parameters
    ----------
    value : object
        Input value (e.g., "None", None, "5", 5).

    Returns
    -------
    int or None
        Parsed integer if provided, otherwise None.

    Raises
    ------
    ValueError
        If `value` is not parseable to an integer or "None".
    """
    if value is None:
        return None

    text = str(value).strip()
    if text.lower() == "none":
        return None

    return int(text)


def parse_float_list(csv_text: Optional[str]) -> Optional[List[float]]:
    """
    Parse a comma-separated list of floats from a string.

    Parameters
    ----------
    csv_text : str or None
        Input string such as "2,1" or "0.7,0.3". If None or empty, returns None.

    Returns
    -------
    list of float or None
        Parsed float values, or None if input is missing/empty.

    Raises
    ------
    ValueError
        If any item cannot be parsed as float.
    """
    if csv_text is None:
        return None

    text = str(csv_text).strip()
    if text == "":
        return None

    return [float(x) for x in text.split(",")]


def upsample_train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Upsample minority classes in the training set to match the majority class.

    This is used to mitigate class imbalance for models where class weights are
    not used (e.g., optional balancing for MLP).

    Parameters
    ----------
    x_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training labels.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    x_balanced : pandas.DataFrame
        Upsampled (balanced) feature matrix.
    y_balanced : pandas.Series
        Upsampled (balanced) labels.
    """
    train_df = x_train.copy()
    train_df["__y__"] = y_train.values

    class_counts = train_df["__y__"].value_counts()
    max_count = int(class_counts.max())

    rng = np.random.default_rng(random_state)
    balanced_parts = []

    for cls, count in class_counts.items():
        class_block = train_df[train_df["__y__"] == cls]
        count = int(count)

        if count < max_count:
            extra_idx = rng.choice(
                class_block.index.to_numpy(),
                size=(max_count - count),
                replace=True,
            )
            class_block = pd.concat([class_block, train_df.loc[extra_idx]], axis=0)

        balanced_parts.append(class_block)

    balanced_df = (
        pd.concat(balanced_parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    y_balanced = balanced_df.pop("__y__")
    x_balanced = balanced_df

    return x_balanced, y_balanced
