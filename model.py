# -*- coding: utf-8 -*-
"""
Model factory functions for the ML pipeline.

This module defines helper functions that construct the supported
machine-learning models with the provided hyperparameters.

Supported models:
- Random Forest
- Histogram-based Gradient Boosting
- Multilayer Perceptron (MLP)
- Soft-voting ensemble (RF + HGB)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier


def build_random_forest(
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_leaf: int,
    max_features: str | int | float,
    class_weight: Optional[str | dict],
    random_state: int,
) -> RandomForestClassifier:
    """
    Construct a RandomForest classifier.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum tree depth. If None, nodes are expanded until all leaves are pure.
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node.
    max_features : str, int, or float
        Number of features to consider when looking for the best split.
    class_weight : str, dict, or None
        Class weighting strategy (e.g., "balanced") or None.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Configured Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=random_state,
    )


def build_hist_gradient_boosting(
    learning_rate: float,
    max_depth: Optional[int],
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    random_state: int,
) -> HistGradientBoostingClassifier:
    """
    Construct a Histogram-based Gradient Boosting classifier.

    Parameters
    ----------
    learning_rate : float
        Learning rate (shrinkage).
    max_depth : int or None
        Maximum tree depth.
    max_leaf_nodes : int
        Maximum number of leaves per tree.
    min_samples_leaf : int
        Minimum number of samples per leaf.
    l2_regularization : float
        L2 regularization parameter.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.ensemble.HistGradientBoostingClassifier
        Configured gradient boosting classifier.
    """
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )


def build_mlp(
    hidden_layer_sizes: Tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
    validation_fraction: float,
    random_state: int,
) -> MLPClassifier:
    """
    Construct a Multilayer Perceptron (MLP) classifier.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int
        Number of neurons in each hidden layer.
    alpha : float
        L2 regularization strength.
    learning_rate_init : float
        Initial learning rate.
    max_iter : int
        Maximum number of training iterations.
    validation_fraction : float
        Fraction of training data used for early stopping validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.neural_network.MLPClassifier
        Configured MLP classifier.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )


def build_ensemble(
    rf: RandomForestClassifier,
    hgb: HistGradientBoostingClassifier,
    weights: List[float],
) -> VotingClassifier:
    """
    Construct a soft-voting ensemble of Random Forest and Gradient Boosting.

    Parameters
    ----------
    rf : RandomForestClassifier
        Pre-configured Random Forest model.
    hgb : HistGradientBoostingClassifier
        Pre-configured Gradient Boosting model.
    weights : list of float
        Voting weights for (rf, hgb).

    Returns
    -------
    sklearn.ensemble.VotingClassifier
        Soft-voting ensemble classifier.
    """
    return VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb)],
        voting="soft",
        weights=weights,
        n_jobs=None,
    )
