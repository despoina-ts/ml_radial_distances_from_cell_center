# ml_radial_distances_from_cell_center

This repository contains the full machine learning pipeline developed as part of my MSc research.  
The project focuses on predicting **radial positioning categories** (central / intermediate / peripheral) of genomic elements based on structural and sequence-derived features.

---

## Project Overview

Spatial genome organization plays a key role in gene regulation and nuclear architecture.  
This project implements a **data-driven machine learning framework** to classify genomic regions into radial distance categories using engineered genomic and epigenomic features.

The pipeline is designed to be:
- reproducible
- modular
- suitable for systematic evaluation of multiple ML models

---

## Input Data

The model expects a **tab-separated (TSV) file** containing:

- Numerical feature columns (e.g. GC content, conservation scores, structural features)
- A target column named `scope` with values:
  - `central`
  - `intermediate`
  - `peripheral`

Certain identifier and coordinate-related columns (gene IDs, genomic start/end positions, bead coordinates, auxiliary scope annotations) are automatically excluded from training.

---

## Machine Learning Pipeline

The script implements the following steps:

1. **Data loading & cleaning**
   - Removal of non-informative identifier columns
   - Handling of missing values (median for numeric, most frequent for categorical)

2. **Feature preprocessing**
   - Optional standardization for neural networks
   - Optional one-hot encoding of chromosome information

3. **Model training**
   Supported models:
   - Random Forest (`rf`)
   - Histogram-based Gradient Boosting (`hgb`)
   - Multilayer Perceptron (`mlp`)
   - Soft-voting ensemble of RF + HGB (`ens`)

4. **Evaluation**
   - Stratified train/test split
   - Accuracy and macro-F1 score
   - Confusion matrix
   - 5-fold stratified cross-validation

5. **Model interpretation & visualization**
   - Feature importance (Random Forest)
   - Correlation heatmaps (Spearman)
   - GC-content vs conservation boxplots across quantiles

---

## Running the Pipeline

### Basic usage (Random Forest)
```bash
python3 ml_pipeline.py --tsv final_feature_table.tsv --model rf
