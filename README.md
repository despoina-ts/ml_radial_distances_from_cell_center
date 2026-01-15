# ml_radial_distances_from_cell_center

This repository contains a modular machine-learning pipeline developed as part of an MSc research project.  
The goal is to predict **radial nuclear positioning categories** — **central**, **intermediate**, and **peripheral** — for genomic elements using sequence- and structure-derived features.

The pipeline combines feature engineering, multiple machine-learning models, rigorous evaluation, and interpretable visualizations.

---

## Scientific background

The spatial organization of the genome inside the nucleus is highly non-random.  
Genes and regulatory regions occupy preferential radial positions from the nuclear center to the periphery, and these positions correlate with transcriptional activity, chromatin state, and regulatory potential.

This project implements a **data-driven classification framework** that learns these spatial patterns from genomic features such as GC content, evolutionary conservation, and structural properties of genes.

---

## Repository Structure

The project is organized into clearly separated scripts and modules.

### scripts/

- **ml_pipeline.py**  
  Thin launcher script that starts the full machine-learning pipeline by calling the command-line interface (CLI).

### src/ml_radial/

- **utils.py**  
  Helper functions for parsing arguments, detecting coordinate columns, dropping identifiers, and upsampling imbalanced training data.

- **preprocessing.py**  
  Builds the feature matrix and target vector.  
  Normalizes scope labels, removes non-informative columns, optionally adds chromosome as a categorical feature, and constructs the sklearn preprocessing pipeline.

- **models.py**  
  Contains factory functions for all machine-learning models:  
  Random Forest, Histogram Gradient Boosting, Multilayer Perceptron, and their soft-voting ensemble.

- **evaluation.py**  
  Computes performance metrics (accuracy, macro-F1, precision, recall, confusion matrix) and performs stratified cross-validation.  
  Saves all metrics in CSV and JSON format.

- **plots.py**  
  Generates all figures used for analysis and interpretation:  
  Spearman correlation heatmaps, Random Forest feature-importance barplots, and GC-content vs conservation boxplots.

- **run_pipeline.py**  
  The pipeline orchestrator.  
  Reads command-line arguments, loads the data, runs preprocessing, trains the selected model, evaluates performance, and saves all outputs.

### Root files

- **requirements.txt / environment.yml**  
  Defines the Python environment required to run the pipeline.

- **README.md**  
  Documentation describing the scientific motivation, pipeline structure, usage, and outputs.

---

## Installation

### Option A — pip + virtual environment

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

### Option B — conda

conda env create -f environment.yml  
conda activate ml_radial  

---

## Input data format

The pipeline expects a **tab-separated (TSV)** file containing:

- Numerical feature columns (e.g. GC content, conservation scores, structural genomic features)  
- A target column named **scope** with values: central, intermediate, peripheral  

Identifier columns (gene IDs, genomic start/end coordinates, bead coordinates, auxiliary scope annotations) are automatically excluded from training.

---

## Running the pipeline

Always run from the repository root:

export PYTHONPATH=src  
python3 scripts/ml_pipeline.py --tsv final_feature_table.tsv --model rf  

---

## Common command-line arguments

--tsv : input TSV file (required)  
--outdir : output directory (default ML_OUT)  
--target : target column name (default scope)  
--model : rf, hgb, mlp, or ens  
--include-chrom : include chromosome as one-hot feature  
--test-size : test fraction (default 0.20)  
--random-state : random seed  

---

## Model-specific arguments

Random Forest (--model rf)  
--n-estimators (default 800)  
--max-depth (default None)  
--min-samples-leaf (default 3)  
--max-features (default sqrt)  
--class-weight (default balanced)  

Histogram Gradient Boosting (--model hgb)  
--hgb-lr (default 0.05)  
--hgb-max-depth (default None)  
--hgb-max-leaf-nodes (default 31)  
--hgb-min-samples-leaf (default 20)  
--hgb-l2 (default 0.0)  

Multilayer Perceptron (--model mlp)  
--mlp-hidden (default 256,128)  
--mlp-alpha (default 5e-4)  
--mlp-lr (default 1e-3)  
--mlp-max-iter (default 200)  
--mlp-val-frac (default 0.1)  
--mlp-balance (upsample classes)  

Ensemble (--model ens)  
--ens-weights (default 2,1)  

---

## Outputs

After a run, ML_OUT contains:

- Trained model (*_scope_model.joblib)  
- Test predictions (test_predictions.tsv)  
- Feature list (features_used.csv)  
- Feature importance (feature_importances_rf.csv)  
- Metrics (metrics.json, model_metrics.csv, confusion_matrix.csv)  
- Figures in figures/ (heatmaps, barplots, GC vs conservation)

---

## Reproducibility

All random operations are controlled by the --random-state argument.  
Given the same input data and seed, the pipeline produces deterministic results.

---

