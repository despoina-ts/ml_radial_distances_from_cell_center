# -*- coding: utf-8 -*-
"""
Plotting utilities for the ML pipeline.

This module provides functions for saving:
- Spearman correlation heatmaps
- Random Forest feature-importance barplots
- GC-content vs Conservation boxplots (GC grouped by conservation quantiles)

All plots are saved using a non-interactive matplotlib backend ("Agg").
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_spearman_corr_heatmap(
    x: pd.DataFrame,
    outdir: str,
    fig_dir: str,
    filename_csv: str = "feature_correlations_spearman.csv",
    filename_png: str = "corr_heatmap_spearman.png",
) -> None:
    """
    Compute and save a Spearman correlation matrix heatmap for numeric features.

    Parameters
    ----------
    x : pandas.DataFrame
        Feature dataframe (only numeric columns are used).
    outdir : str
        Output directory for saving the correlation matrix CSV.
    fig_dir : str
        Output directory for saving figures.
    filename_csv : str, default="feature_correlations_spearman.csv"
        Filename for the CSV correlation matrix.
    filename_png : str, default="corr_heatmap_spearman.png"
        Filename for the PNG heatmap.
    """
    corr = x.select_dtypes(include=[np.number]).corr(method="spearman")
    corr.to_csv(os.path.join(outdir, filename_csv))

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr.values, aspect="auto")
    plt.title("Spearman Correlation (numeric features)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(
        ticks=np.arange(len(corr.columns)),
        labels=corr.columns,
        rotation=90,
    )
    plt.yticks(
        ticks=np.arange(len(corr.index)),
        labels=corr.index,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename_png), dpi=200)
    plt.close()


def save_rf_importance_barplot(
    rf_importances: pd.DataFrame,
    fig_dir: str,
    top_k: int = 20,
    filename_prefix: str = "barplot_feature_importance_rf_top",
) -> None:
    """
    Save a horizontal barplot of Random Forest feature importances.

    Parameters
    ----------
    rf_importances : pandas.DataFrame
        DataFrame with columns: ["feature", "importance"], sorted descending.
    fig_dir : str
        Output directory for saving figures.
    top_k : int, default=20
        Number of top features to plot.
    filename_prefix : str, default="barplot_feature_importance_rf_top"
        Prefix for output PNG filename.
    """
    if rf_importances is None or rf_importances.empty:
        return

    top_k = min(top_k, len(rf_importances))
    top_df = rf_importances.head(top_k).iloc[::-1]

    plt.figure(figsize=(9, max(4, 0.35 * top_k + 1)))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance")
    plt.title(f"Random Forest Feature Importance (Top {top_k})")
    plt.tight_layout()

    outname = f"{filename_prefix}{top_k}.png"
    plt.savefig(os.path.join(fig_dir, outname), dpi=200)
    plt.close()


def save_gc_conservation_boxplots(
    df: pd.DataFrame,
    fig_dir: str,
    q: int = 5,
    filename_png: str = "ALL_THE_BOXES_TOGETHER.png",
) -> None:
    """
    Create a multi-panel figure of GC-content vs Conservation boxplots.

    For each GC column (gc_content_gene/exon/intron) and each Conservation column
    (conservation_gene/exon/intron), conservation values are binned into
    quantiles (default: 5). GC values are then shown as boxplots across
    conservation quantiles.

    Parameters
    ----------
    df : pandas.DataFrame
        Original input dataframe containing GC and conservation columns.
    fig_dir : str
        Output directory for saving figures.
    q : int, default=5
        Number of quantile bins for conservation.
    filename_png : str, default="ALL_THE_BOXES_TOGETHER.png"
        Output filename for the multi-panel figure.
    """
    gc_names = {"gc_content_gene", "gc_content_exon", "gc_content_intron"}
    cons_names = {"conservation_gene", "conservation_exon", "conservation_intron"}

    gc_cols = [c for c in df.columns if c.lower() in gc_names]
    cons_cols = [c for c in df.columns if c.lower() in cons_names]

    # Fallback for uncommon capitalization (kept for compatibility)
    if not gc_cols:
        for name in ["gc_content_gene", "gc_content_exon", "gc_content_intron"]:
            if name in df.columns:
                gc_cols.append(name)

    if not cons_cols:
        for name in ["conservation_gene", "conservation_exon", "conservation_intron"]:
            if name in df.columns:
                cons_cols.append(name)

    pairs = [(g, c) for g in gc_cols for c in cons_cols]
    if len(pairs) == 0:
        print("[info] No GC×Conservation pairs found. Skipping ALL_THE_BOXES_TOGETHER.")
        return

    n_panels = len(pairs)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    def quantile_bins(series: pd.Series, q_bins: int) -> tuple[pd.Series, list[str]]:
        s = pd.to_numeric(series, errors="coerce")
        qcut = pd.qcut(s, q=q_bins, duplicates="drop")
        labels = [f"Q{i + 1}" for i in range(len(qcut.cat.categories))]
        return qcut, labels

    panel_idx = 0
    for gc_col, cons_col in pairs:
        r_i = panel_idx // n_cols
        c_i = panel_idx % n_cols
        ax = axes[r_i, c_i]

        gc_vals = pd.to_numeric(df[gc_col], errors="coerce")
        cons_vals = pd.to_numeric(df[cons_col], errors="coerce")

        mask = np.isfinite(gc_vals) & np.isfinite(cons_vals)
        gc_vals = gc_vals[mask]
        cons_vals = cons_vals[mask]

        if len(gc_vals) < 10:
            ax.set_axis_off()
            panel_idx += 1
            continue

        qbins, qlabels = quantile_bins(cons_vals, q_bins=q)

        data = [
            gc_vals[qbins == cat].dropna().values for cat in qbins.cat.categories
        ]

        ax.boxplot(data, labels=qlabels, showfliers=True)
        ax.set_xlabel(f"{cons_col} (quantiles low→high)")
        ax.set_ylabel(gc_col)
        ax.set_title(f"{gc_col} vs {cons_col}", fontsize=10)

        panel_idx += 1

    # Hide any unused axes
    for k in range(panel_idx, n_rows * n_cols):
        r_i = k // n_cols
        c_i = k % n_cols
        axes[r_i, c_i].set_axis_off()

    plt.suptitle(
        "ALL THE BOXES TOGETHER — GC content vs Conservation (GC by conservation quantiles)",
        y=0.995,
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    outpath = os.path.join(fig_dir, filename_png)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[ok] Saved: {outpath}")
