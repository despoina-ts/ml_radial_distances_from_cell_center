#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re, json
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def is_start_end(col: str) -> bool:
    c = col.lower().replace(" ", "").replace("-", "_")
    return bool(re.search(r'(gene|tx|transcript).*(start|end)', c))

def default_drop_cols(cols):
    drop = set()
    for c in cols:
        cl = c.lower()
        if is_start_end(c): drop.add(c)
        if cl in {
            "gene.stable.id","gene.name","bead_id","bead_start","bead_end",
            "scope","scope_hard","scope_balanced","scope_partial","assign_note",
            "overlap_bp","dist_zero_norm"
        }: drop.add(c)
    return list(drop)

def build_preprocessor(numeric_cols, cat_cols, scale_numeric=False):
    transformers=[]
    if numeric_cols:
        if scale_numeric:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), numeric_cols))
        else:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def upsample_train(X_train, y_train, random_state=42):
    tr = X_train.copy(); tr["__y__"] = y_train.values
    vc = tr["__y__"].value_counts()
    n_max = vc.max()
    rng = np.random.default_rng(random_state)
    parts = []
    for cls, n in vc.items():
        block = tr[tr["__y__"] == cls]
        if n < n_max:
            take = rng.choice(block.index.to_numpy(), size=(n_max - n), replace=True)
            block = pd.concat([block, tr.loc[take]], axis=0)
        parts.append(block)
    tr_bal = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state)
    yb = tr_bal.pop("__y__")
    Xb = tr_bal
    return Xb, yb

def parse_optional_int(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() == "none":
        return None
    return int(s)

def parse_float_list(csv_text):
    if csv_text is None or str(csv_text).strip()=="":
        return None
    return [float(x) for x in str(csv_text).split(",")]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Predict scope (central/intermediate/peripheral) with RF / HGB / MLP / ENS + plots.")
    ap.add_argument("--tsv", required=True, help="Final TSV with target column (default 'scope').")
    ap.add_argument("--outdir", default="ML_OUT")
    ap.add_argument("--target", default="scope")
    ap.add_argument("--model", choices=["rf","hgb","mlp","ens"], default="rf")
    ap.add_argument("--include-chrom", dest="include_chrom", action="store_true",
                    help="Include Chromosome.scaffold.name as one-hot feature if present.")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=42)

    # RF params
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=3)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--class-weight", default="balanced")

    # HGB params
    ap.add_argument("--hgb-lr", type=float, default=0.05)
    ap.add_argument("--hgb-max-depth", default="None")
    ap.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    ap.add_argument("--hgb-min-samples-leaf", type=int, default=20)
    ap.add_argument("--hgb-l2", type=float, default=0.0)

    # MLP params
    ap.add_argument("--mlp-hidden", type=str, default="256,128")
    ap.add_argument("--mlp-alpha", type=float, default=5e-4)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-max-iter", type=int, default=200)
    ap.add_argument("--mlp-val-frac", type=float, default=0.1)
    ap.add_argument("--mlp-balance", action="store_true", help="Upsample classes in TRAIN for MLP.")

    # ENS params
    ap.add_argument("--ens-weights", type=str, default="2,1",
                    help="Comma-separated weights for (RF,HGB) in soft voting, e.g. '2,1'.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fig_dir = os.path.join(args.outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    try:
        args.hgb_max_depth = parse_optional_int(args.hgb_max_depth)
    except Exception as e:
        raise SystemExit(f" --hgb-max-depth must be integer or 'None'. Got: {args.hgb_max_depth}") from e

    # ---- load & target ----
    df = pd.read_csv(args.tsv, sep="\t", engine="python")
    if args.target not in df.columns:
        raise SystemExit(f" Target '{args.target}' not found in {args.tsv}")
    y = (df[args.target].astype(str).str.strip().str.lower()
         .replace({"centre":"central","center":"central","periphery":"peripheral","inter":"intermediate"}))

    # ---- features ----
    drop_cols = default_drop_cols(df.columns)
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    chrom_col = None
    for cand in ["Chromosome.scaffold.name","chrom","Chromosome","chr","CHR"]:
        if cand in df.columns:
            chrom_col = cand; break

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = []
    if args.include_chrom and chrom_col is not None:
        if chrom_col not in X.columns:
            X[chrom_col] = df[chrom_col]
        cat_cols.append(chrom_col)

    other_obj = [c for c in X.select_dtypes(include=["object","string"]).columns if c not in cat_cols]
    if other_obj:
        X = X.drop(columns=other_obj)

    valid = y.isin(["central", "intermediate", "peripheral"])
    X = X.loc[valid].copy(); y = y.loc[valid].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    if args.model == "mlp":
        pre = build_preprocessor(numeric_cols, cat_cols, scale_numeric=True)
    else:
        pre = build_preprocessor(numeric_cols, cat_cols, scale_numeric=False)

    # -------- train chosen model --------
    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_jobs=-1,
            class_weight=(None if str(args.class_weight).lower()=="none" else args.class_weight),
            random_state=args.random_state
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

    elif args.model == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=args.hgb_lr,
            max_depth=args.hgb_max_depth,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            min_samples_leaf=args.hgb_min_samples_leaf,
            l2_regularization=args.hgb_l2,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=args.random_state
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

    elif args.model == "mlp":
        hidden = tuple(int(v) for v in args.mlp_hidden.split(",")) if args.mlp_hidden else (256,128)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=args.mlp_alpha,
            learning_rate_init=args.mlp_lr,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=args.mlp_val_frac,
            random_state=args.random_state
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        if args.mlp_balance:
            Xb, yb = upsample_train(X_train, y_train, random_state=args.random_state)
            pipe.fit(Xb, yb)
        else:
            pipe.fit(X_train, y_train)

    else:  # ens
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_jobs=-1,
            class_weight=(None if str(args.class_weight).lower()=="none" else args.class_weight),
            random_state=args.random_state
        )
        hgb = HistGradientBoostingClassifier(
            learning_rate=args.hgb_lr,
            max_depth=args.hgb_max_depth,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            min_samples_leaf=args.hgb_min_samples_leaf,
            l2_regularization=args.hgb_l2,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=args.random_state
        )
        w = parse_float_list(args.ens_weights) or [2.0, 1.0]
        if len(w) != 2:
            raise SystemExit(" --ens-weights must have exactly 2 numbers, e.g. '2,1'.")
        ens = VotingClassifier(
            estimators=[("rf", rf), ("hgb", hgb)],
            voting="soft",
            weights=w,
            n_jobs=None
        )
        pipe = Pipeline([("pre", pre), ("clf", ens)])
        pipe.fit(X_train, y_train)

    # -------- evaluation --------
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    rep = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=["central","intermediate","peripheral"])

    print("\n=== TEST METRICS ===")
    print("Model:", args.model.upper())
    print("Accuracy:", round(acc,4))
    print("Macro-F1:", round(f1m,4))
    print(rep)
    print("Confusion matrix [rows=true; cols=pred] (central, intermediate, peripheral):\n", cm)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(pipe, X, y, scoring="f1_macro", cv=cv, n_jobs=-1)
    print("5-fold macro-F1:", np.round(cv_scores,4), " | mean:", round(cv_scores.mean(),4))

    # -------- save artifacts --------
    model_name = f"{args.model}_scope_model.joblib"
    joblib.dump(pipe, os.path.join(args.outdir, model_name))

    pred_df = X_test.copy()
    pred_df["true_scope"] = y_test.values
    pred_df["pred_scope"] = y_pred
    pred_df.to_csv(os.path.join(args.outdir, "test_predictions.tsv"), sep="\t", index=False)

    # feature names used
    feat_names = []
    if numeric_cols: feat_names += numeric_cols
    if cat_cols:
        oh = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
        feat_names += oh.get_feature_names_out(cat_cols).tolist()
    pd.Series(feat_names, name="feature").to_csv(
        os.path.join(args.outdir, "features_used.csv"), index=False
    )

    # RF importances (if RF or ENS with RF)
    rf_importances = None
    try:
        if args.model == "rf":
            imps = pipe.named_steps["clf"].feature_importances_
            if len(imps)==len(feat_names):
                rf_importances = pd.DataFrame({"feature":feat_names,"importance":imps})\
                                  .sort_values("importance",ascending=False)
                rf_importances.to_csv(os.path.join(args.outdir,"feature_importances_rf.csv"), index=False)
        elif args.model == "ens":
            est = pipe.named_steps["clf"].estimators_[0]
            imps = est.feature_importances_
            if len(imps)==len(feat_names):
                rf_importances = pd.DataFrame({"feature":feat_names,"importance":imps})\
                                  .sort_values("importance",ascending=False)
                rf_importances.to_csv(os.path.join(args.outdir,"feature_importances_rf.csv"), index=False)
    except Exception:
        pass

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({
            "model": args.model,
            "test_accuracy": float(acc),
            "test_macro_f1": float(f1m),
            "labels": ["central","intermediate","peripheral"],
            "confusion_matrix": cm.tolist(),
            "cv_macro_f1_mean": float(cv_scores.mean()),
            "cv_macro_f1_scores": [float(x) for x in cv_scores]
        }, f, indent=2)

    # === evaluation metrics CSVs ===
    labels_fixed = ["central","intermediate","peripheral"]
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=labels_fixed, zero_division=0)
    rows = [
        {"metric":"accuracy","class":"","value":float(acc)},
        {"metric":"macro_f1","class":"","value":float(f1m)},
    ]
    for i,lab in enumerate(labels_fixed):
        rows += [
            {"metric":"precision","class":lab,"value":float(pr[i])},
            {"metric":"recall","class":lab,"value":float(rc[i])},
            {"metric":"f1","class":lab,"value":float(f1[i])},
            {"metric":"support","class":lab,"value":int(sup[i])},
        ]
    cm_tbl = pd.DataFrame(cm, index=[f"true_{c}" for c in labels_fixed],
                             columns=[f"pred_{c}" for c in labels_fixed])
    cm_tbl.to_csv(os.path.join(args.outdir, "confusion_matrix.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "model_metrics.csv"), index=False)

    # --------- OTHER PLOTS ---------
    # Spearman heatmap
    try:
        corr = X.select_dtypes(include=[np.number]).corr(method="spearman")
        corr.to_csv(os.path.join(args.outdir, "feature_correlations_spearman.csv"))
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr.values, aspect="auto")
        plt.title("Spearman Correlation (numeric features)")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
        plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "corr_heatmap_spearman.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"[warn] Correlation heatmap failed: {e}")

    # Feature importance barh (if RF/ENS with RF)
    if rf_importances is not None and not rf_importances.empty:
        try:
            top_k = min(20, len(rf_importances))
            top_df = rf_importances.head(top_k).iloc[::-1]
            plt.figure(figsize=(9, max(4, 0.35*top_k + 1)))
            plt.barh(top_df["feature"], top_df["importance"])
            plt.xlabel("Importance")
            plt.title("Random Forest Feature Importance (Top {})".format(top_k))
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "barplot_feature_importance_rf_top{}.png".format(top_k)), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[warn] Could not create RF feature-importance barplot: {e}")

    # -------- *** GC × Conservation BOX-PLOTS IN ONE FIGURE *** --------
    try:
        # Identify GC and Conservation columns present
        gc_cols   = [c for c in df.columns if c.lower() in {"gc_content_gene","gc_content_exon","gc_content_intron"}]
        cons_cols = [c for c in df.columns if c.lower() in {"conservation_gene","conservation_exon","conservation_intron"}]

        # Fallback in case of different capitalization
        if not gc_cols:
            for name in ["gc_content_gene","gc_content_exon","gc_content_intron"]:
                if name in df.columns: gc_cols.append(name)
        if not cons_cols:
            for name in ["conservation_gene","conservation_exon","conservation_intron"]:
                if name in df.columns: cons_cols.append(name)

        # Build all GC × Conservation pairs (up to 9)
        pairs = [(g, c) for g in gc_cols for c in cons_cols]

        if len(pairs) == 0:
            print("[info] No GC×Conservation pairs found. Skipping ALL_THE_BOXES_TOGETHER.")
        else:
            # Number of panels: up to len(pairs); grid is dynamic (≤ 3 columns)
            n = len(pairs)
            cols = 3
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.2*rows))
            axes = np.array(axes).reshape(rows, cols)

            # function: make quantile bins labels for conservation
            def qbin_labels(series, q=5):
                s = pd.to_numeric(series, errors="coerce")
                qcut = pd.qcut(s, q=q, duplicates="drop")
                # Build clean labels like Q1 … Qk
                levels = qcut.cat.categories
                labels = [f"Q{i+1}" for i in range(len(levels))]
                return qcut, labels

            panel_idx = 0
            for gcol, ccol in pairs:
                r_i = panel_idx // cols
                c_i = panel_idx % cols
                ax = axes[r_i, c_i]

                # Prepare data
                g = pd.to_numeric(df[gcol], errors="coerce")
                c = pd.to_numeric(df[ccol], errors="coerce")
                m = np.isfinite(g) & np.isfinite(c)
                g = g[m]; c = c[m]

                if len(g) < 10:
                    ax.set_axis_off()
                    panel_idx += 1
                    continue

                # Bin conservation into quantiles (Q=5 by default)
                qbins, qlabels = qbin_labels(c, q=5)

                # Build data lists per bin (GC values per conservation-quantile)
                data = [g[qbins == cat].dropna().values for cat in qbins.cat.categories]

                # Boxplot with outliers (as in your example)
                ax.boxplot(data, labels=qlabels, showfliers=True)
                ax.set_xlabel(f"{ccol} (quantiles low→high)")
                ax.set_ylabel(gcol)
                ax.set_title(f"{gcol} vs {ccol}", fontsize=10)

                panel_idx += 1

            # Hide any empty axes
            for k in range(panel_idx, rows*cols):
                r_i = k // cols; c_i = k % cols
                axes[r_i, c_i].set_axis_off()

            plt.suptitle("ALL THE BOXES TOGETHER — GC content vs Conservation (GC by conservation quantiles)", y=0.995, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            outpath = os.path.join(fig_dir, "ALL_THE_BOXES_TOGETHER.png")
            plt.savefig(outpath, dpi=200)
            plt.close()
            print(f"[ok] Saved: {outpath}")
    except Exception as e:
        print(f"[warn] GC×Conservation boxplots failed: {e}")

    print(f"\n Figures saved in: {fig_dir}")

if __name__ == "__main__":
    main()
