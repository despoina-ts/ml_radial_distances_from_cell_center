# -*- coding: utf-8 -*-
"""
Command-line interface for the ML pipeline.

This module connects preprocessing, model construction, training,
evaluation, and plotting into a single executable workflow.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .utils import parse_float_list, parse_optional_int, upsample_train
from .preprocessing import prepare_features_and_target, build_preprocessor
from .models import (
    build_random_forest,
    build_hist_gradient_boosting,
    build_mlp,
    build_ensemble,
)
from .evaluation import (
    evaluate_on_test,
    cross_validate_macro_f1,
    save_metrics_json,
    save_metrics_csvs,
)
from .plots import (
    save_spearman_corr_heatmap,
    save_rf_importance_barplot,
    save_gc_conservation_boxplots,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    ap = argparse.ArgumentParser(
        description="Predict scope (central/intermediate/peripheral) using ML models."
    )

    ap.add_argument("--tsv", required=True, help="Input TSV file containing features and target.")
    ap.add_argument("--outdir", default="ML_OUT", help="Output directory.")
    ap.add_argument("--target", default="scope", help="Target column name.")
    ap.add_argument("--model", choices=["rf", "hgb", "mlp", "ens"], default="rf")
    ap.add_argument(
        "--include-chrom",
        action="store_true",
        help="Include chromosome as a one-hot encoded categorical feature.",
    )
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=42)

    # Random Forest
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=3)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--class-weight", default="balanced")

    # HGB
    ap.add_argument("--hgb-lr", type=float, default=0.05)
    ap.add_argument("--hgb-max-depth", default="None")
    ap.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    ap.add_argument("--hgb-min-samples-leaf", type=int, default=20)
    ap.add_argument("--hgb-l2", type=float, default=0.0)

    # MLP
    ap.add_argument("--mlp-hidden", default="256,128")
    ap.add_argument("--mlp-alpha", type=float, default=5e-4)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-max-iter", type=int, default=200)
    ap.add_argument("--mlp-val-frac", type=float, default=0.1)
    ap.add_argument("--mlp-balance", action="store_true")

    # Ensemble
    ap.add_argument("--ens-weights", default="2,1")

    return ap


def main() -> None:
    """
    Run the full machine-learning pipeline from the command line.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fig_dir = os.path.join(args.outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    try:
        args.hgb_max_depth = parse_optional_int(args.hgb_max_depth)
    except Exception as exc:
        raise SystemExit(f"--hgb-max-depth must be integer or 'None'. Got: {args.hgb_max_depth}") from exc

    df = pd.read_csv(args.tsv, sep="\t")

    x, y, numeric_cols, cat_cols, _ = prepare_features_and_target(
        df, target_col=args.target, include_chrom=args.include_chrom
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        scale_numeric=(args.model == "mlp"),
    )

    class_weight = None if str(args.class_weight).lower() == "none" else args.class_weight

    if args.model == "rf":
        clf = build_random_forest(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight=class_weight,
            random_state=args.random_state,
        )

    elif args.model == "hgb":
        clf = build_hist_gradient_boosting(
            learning_rate=args.hgb_lr,
            max_depth=args.hgb_max_depth,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            min_samples_leaf=args.hgb_min_samples_leaf,
            l2_regularization=args.hgb_l2,
            random_state=args.random_state,
        )

    elif args.model == "mlp":
        hidden = tuple(int(v) for v in args.mlp_hidden.split(","))
        clf = build_mlp(
            hidden_layer_sizes=hidden,
            alpha=args.mlp_alpha,
            learning_rate_init=args.mlp_lr,
            max_iter=args.mlp_max_iter,
            validation_fraction=args.mlp_val_frac,
            random_state=args.random_state,
        )

    else:  # ens
        rf = build_random_forest(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight=class_weight,
            random_state=args.random_state,
        )
        hgb = build_hist_gradient_boosting(
            learning_rate=args.hgb_lr,
            max_depth=args.hgb_max_depth,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            min_samples_leaf=args.hgb_min_samples_leaf,
            l2_regularization=args.hgb_l2,
            random_state=args.random_state,
        )
        weights = parse_float_list(args.ens_weights) or [2.0, 1.0]
        clf = build_ensemble(rf, hgb, weights)

    pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])

    if args.model == "mlp" and args.mlp_balance:
        x_train, y_train = upsample_train(x_train, y_train, random_state=args.random_state)

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    metrics = evaluate_on_test(y_test, y_pred)

    print("\n=== TEST METRICS ===")
    print("Model:", args.model.upper())
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Macro-F1:", round(metrics["macro_f1"], 4))
    print(metrics["report"])
    print("Confusion matrix:\n", metrics["confusion_matrix"])

    cv_scores = cross_validate_macro_f1(pipeline, x, y, args.random_state)
    print("5-fold macro-F1:", np.round(cv_scores, 4), " | mean:", round(cv_scores.mean(), 4))

    joblib.dump(pipeline, os.path.join(args.outdir, f"{args.model}_scope_model.joblib"))

    preds = x_test.copy()
    preds["true_scope"] = y_test.values
    preds["pred_scope"] = y_pred
    preds.to_csv(os.path.join(args.outdir, "test_predictions.tsv"), sep="\t", index=False)

    save_metrics_json(args.outdir, args.model, metrics, cv_scores)
    save_metrics_csvs(args.outdir, y_test, y_pred, metrics)

    try:
        save_spearman_corr_heatmap(x, args.outdir, fig_dir)
    except Exception as exc:
        print(f"[warn] Correlation heatmap failed: {exc}")

    try:
        if args.model in {"rf", "ens"}:
            if args.model == "rf":
                imps = clf.feature_importances_
            else:
                imps = clf.estimators_[0].feature_importances_

            feature_names = (
                numeric_cols
                + (
                    pipeline.named_steps["pre"]
                    .named_transformers_["cat"]
                    .named_steps["oh"]
                    .get_feature_names_out(cat_cols).tolist()
                    if cat_cols
                    else []
                )
            )

            if len(imps) == len(feature_names):
                importances_df = pd.DataFrame(
                    {"feature": feature_names, "importance": imps}
                ).sort_values("importance", ascending=False)

                importances_df.to_csv(
                    os.path.join(args.outdir, "feature_importances_rf.csv"),
                    index=False,
                )

                save_rf_importance_barplot(importances_df, fig_dir)

    except Exception as exc:
        print(f"[warn] RF importance plotting failed: {exc}")

    try:
        save_gc_conservation_boxplots(df, fig_dir)
    except Exception as exc:
        print(f"[warn] GC×Conservation plots failed: {exc}")

    print(f"\nFigures saved in: {fig_dir}")
