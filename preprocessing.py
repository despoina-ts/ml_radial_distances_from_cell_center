# -*- coding: utf-8 -*-
"""
Preprocessing utilities for feature/target preparation.

This module handles:
- Target label normalization
- Feature selection and column dropping
- Chromosome column detection (optional categorical feature)
- Building the sklearn ColumnTransformer preprocessor
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import default_drop_cols


def normalize_scope_labels(y: pd.Series) -> pd.Series:
    """
    Normalize scope labels to a consistent set of values.

    The function standardizes common variants into:
    - "central"
    - "intermediate"
    - "peripheral"

    Parameters
    ----------
    y : pandas.Series
        Raw target labels.

    Returns
    -------
    pandas.Series
        Normalized target labels (lowercase strings).
    """
    return (
        y.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "centre": "central",
                "center": "central",
                "periphery": "peripheral",
                "inter": "intermediate",
            }
        )
    )


def resolve_chrom_col(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a chromosome column from common candidate names.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    str or None
        Name of the detected chromosome column, or None if not found.
    """
    candidates = ["Chromosome.scaffold.name", "chrom", "Chromosome", "chr", "CHR"]
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def build_preprocessor(
    numeric_cols: List[str],
    cat_cols: List[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Build an sklearn ColumnTransformer for numeric + categorical preprocessing.

    Numeric preprocessing:
    - median imputation
    - optional StandardScaler (recommended for MLP)

    Categorical preprocessing:
    - most-frequent imputation
    - one-hot encoding with unknown handling

    Parameters
    ----------
    numeric_cols : list of str
        Names of numeric feature columns.
    cat_cols : list of str
        Names of categorical feature columns.
    scale_numeric : bool, default=False
        If True, applies StandardScaler to numeric features.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocessing transformer suitable for sklearn Pipelines.
    """
    transformers = []

    if numeric_cols:
        if scale_numeric:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            )
        else:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "scope",
    include_chrom: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], Optional[str]]:
    """
    Prepare the feature matrix and target vector for classification.

    Steps performed:
    - Validate presence of the target column.
    - Normalize target labels to {central, intermediate, peripheral}.
    - Drop identifier/coordinate/helper columns using default rules.
    - Optionally include a chromosome column as a categorical feature.
    - Drop other object/string columns (to avoid accidental leakage).
    - Filter out rows with invalid target labels.
    - Identify numeric and categorical feature columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features and the target column.
    target_col : str, default="scope"
        Name of the target column.
    include_chrom : bool, default=False
        If True, includes an available chromosome column as one-hot encoded
        categorical feature.

    Returns
    -------
    x : pandas.DataFrame
        Cleaned feature matrix.
    y : pandas.Series
        Normalized target labels.
    numeric_cols : list of str
        Numeric feature column names detected in `x`.
    cat_cols : list of str
        Categorical feature column names included in `x`.
    chrom_col : str or None
        Detected chromosome column name (None if not found).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = normalize_scope_labels(df[target_col])

    drop_cols = default_drop_cols(df.columns)
    x = df.drop(columns=drop_cols, errors="ignore").copy()

    chrom_col = resolve_chrom_col(df)
    cat_cols: List[str] = []

    if include_chrom and chrom_col is not None:
        if chrom_col not in x.columns:
            x[chrom_col] = df[chrom_col]
        cat_cols.append(chrom_col)

    # Drop any remaining object/string columns not explicitly allowed
    other_obj_cols = [
        c
        for c in x.select_dtypes(include=["object", "string"]).columns
        if c not in cat_cols
    ]
    if other_obj_cols:
        x = x.drop(columns=other_obj_cols)

    # Filter to valid labels only
    valid_mask = y.isin(["central", "intermediate", "peripheral"])
    x = x.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()

    return x, y, numeric_cols, cat_cols, chrom_col
