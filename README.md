# Age-pred-multi

A two-level multimodal machine learning pipeline for predicting chronological age (`Age`), cognitive performance (`Cog`), or an equal-weight composite of both (`Mix`) from neuroimaging, electrophysiological, and behavioral features, followed by feature-block ablation, predicted age difference (PAD) correlation analysis, and publication-ready visualization.

## Overview

The pipeline integrates multiple brain and behavioral modalities through a two-level stacking architecture:

1. **Level 1 (Modality-Specific Models)**: Fits per-modality cross-validated models on each feature block (`DTI_FA`, `DTI_MD`, `rs-EEG_PSD`, `rs-MRI_FC`, `MRI_ROI`, `FUN_COGNITIVE`, `BEH_COGNITIVE`) and incorporates pre-trained T1w brain-age predictions (`Pyment`). Level 1 supports two output routes:

   * **`targ-preds`**: Passes out-of-fold / test target predictions from each modality to Level 2.

   * **`cross-decomp`**: Projects each modality onto supervised PLS latent components (with fixed $K$, inner-CV selection, or cross-validated permutation Parallel Analysis `--pls_components auto`, sign-aligned across folds).

2. **Level 2 (Multimodal Stacking Model)**: Trains a cross-validated second-level regressor (meta-model, `--model_lv2`, with median imputation for missing modalities) on the merged Level-1 predictions or PLS components to produce the final multimodal predictions, applies Cole's linear age-bias correction per fold, and (for `Mix` targets) maps composite predictions back to `Age` and `Cog` scales.

3. **Downstream Evaluation**: Evaluates modality importance via multi-seed ablation and stepwise selection (`eval_feats_imp.py`), summarizes fits and feature weights (`summ_results.py`), computes FDR-corrected PAD–cognition correlations (`calc_corr.py`), and generates figures (`plot_results.py`).

## Repository Structure

```
Age-pred-multi/
├── etc/
│   ├── pyment_config.json
│   ├── pyment_preprocess.Dockerfile
│   └── pyment_preprocess_and_finetune.Dockerfile
├── scripts/
│   ├── fetch_dropbox_files.py       # Download raw/preprocessed source files
│   ├── run_tbss.sh                  # FSL TBSS processing for DTI FA/MD stacks
│   ├── prep_pyment.py               # Prepare T1w inputs for Pyment
│   ├── run_pyment.sh                # Run containerized Pyment brain-age prediction
│   ├── make_df_preproc.py           # Build df_preproc.csv & preproc_features.txt
│   ├── make_df_scores.py            # Build cognitive score tables & cog_scores.json
│   ├── make_df_dti_roi-means.py     # Extract DTI ROI mean features
│   ├── make_df_rs-eeg_psd.py        # Extract resting-state EEG PSD features (MNE)
│   ├── make_df_rs-fmri_fc.py        # Extract resting-state fMRI FC features (Nilearn)
│   ├── models.py                    # Custom estimators: TargetScaler, MSVR, MOGP (ICM)
│   ├── training.py                  # Pipelines, CV, Optuna XGBoost tuning, PLS Parallel Analysis
│   ├── helpers.py                   # NIfTI/TBSS caching, CSV/NPZ I/O, run lookup utilities
│   ├── plotting.py                  # Matplotlib/Seaborn figure primitives & Style helper
│   ├── utils.py                     # Logging (custom_print, tee_output, mute_print), JSON helpers
│   ├── predict_ages.py              # Main two-level training & prediction pipeline
│   ├── eval_feats_imp.py            # Multi-seed modality ablation & stepwise evaluation
│   ├── summ_results.py              # Regression fit metrics & Level-2 coefficient extraction
│   ├── calc_corr.py                 # PAD x cognitive score (partial) correlations & BH-FDR
│   └── plot_results.py              # Generate scatter, weight/importance, MAE, box & heatmap plots
├── environment.yml                  # Pinned Conda environment (conda-forge, Python 3.12)
├── environment.txt                  # Exact linux-64 Conda @EXPLICIT lockfile
└── .gitignore

```

Runtime outputs are written outside `scripts/` under `data/`, `models/`, `results/`, and `figures/`.

## Installation

Create the Conda environment from `environment.yml`:

```
conda env create -f environment.yml
conda activate age_pred

```

For an exact bit-for-bit reproduction on `linux-64`, use the explicit lockfile:

```
conda create --name age_pred --file environment.txt
conda activate age_pred

```

**Notes**:

* `py-xgboost` is installed with the CUDA build and automatically falls back to CPU when no CUDA GPU is available (or when `--xgb_device cpu` is passed).

* `torch` and `gpytorch` are imported lazily inside `models.py` and are only required when using `--model_lv1 mogp` or `--model_lv2 mogp`.

* Dockerfiles for running Pyment preprocessing and inference are provided in `etc/`.

## Supported Modalities & Models

### Feature Blocks

| Block | Modality / Description | Source | 
| ----- | ----- | ----- | 
| `DTI_FA` | 4-D TBSS Fractional Anisotropy (downsampled + masked, cached as `.npy`) | `run_tbss.sh` | 
| `DTI_MD` | 4-D TBSS Mean Diffusivity (downsampled + masked, cached as `.npy`) | `run_tbss.sh` | 
| `rs-EEG_PSD` | Resting-state EEG power spectral density | `make_df_rs-eeg_psd.py` | 
| `rs-MRI_FC` | Resting-state fMRI functional connectivity | `make_df_rs-fmri_fc.py` | 
| `MRI_ROI` | Structural T1w MRI ROI features (`STRUCTURE*`, excluding FA) | `make_df_preproc.py` | 
| `FUN_COGNITIVE` | Task-based MRI / EEG cognitive features (`LANGUAGE`, `MEMORY`, `MOTOR`) | `make_df_preproc.py` | 
| `BEH_COGNITIVE` | Behavioral cognitive features (`LANGUAGE`, `MEMORY`, `MOTOR`) | `make_df_preproc.py` | 
| `Pyment` | Pre-trained 3-D CNN brain-age predictions from T1w MRI | `run_pyment.sh` | 

### Regression Algorithms (`--model_lv1` / `--model_lv2`)

| Key | Model | Hyperparameter Tuning | 
| ----- | ----- | ----- | 
| `elasticnet` | `ElasticNetCV` / `MultiTaskElasticNetCV` | Inner 5-fold CV over `L1_RATIOS` × `ALPHAS` | 
| `lasso` | `LassoCV` / `MultiTaskLassoCV` | Inner 5-fold CV over `ALPHAS` | 
| `ridge` | `RidgeCV` | Inner 5-fold CV over `ALPHAS` | 
| `xgboost` | `XGBRegressor` (CPU/CUDA with OOM fallback) | Optuna TPE (`--opt_trials`) | 
| `msvr` | Custom RBF Multi-Output SVR (joint-intercept IRWLS) | Inner 5-fold `GridSearchCV` over `C`, `epsilon`, `gamma` | 
| `mogp` | Custom GPyTorch ICM Multi-Output GP | Exact Marginal Log-Likelihood (Adam) | 
| `pls` | `PLSRegression` | Fixed `K`, inner 5-fold CV (`cv`), or OOF Parallel Analysis (`auto`) | 

## Usage

Run all scripts from the repository root (or inside `scripts/`).

### 1. Data Preparation

Prepare tabular features, cognitive scores, TBSS volumes, and Pyment predictions:

```
python scripts/make_df_preproc.py
python scripts/make_df_scores.py
python scripts/make_df_rs-eeg_psd.py
python scripts/make_df_rs-fmri_fc.py
bash scripts/run_tbss.sh
bash scripts/run_pyment.sh

```

### 2. Two-Level Training & Prediction (`predict_ages.py`)

Target selection via `--targets`: `0` (`[Age]`), `1` (`[Age, Cog]`, default), `2` (`[Mix]`).

```
# Default: Age + Cog, ElasticNet (Lv1) -> Ridge (Lv2)
python scripts/predict_ages.py --targets 1 --model_lv1 elasticnet --model_lv2 ridge

# Equal-weight composite target (Mix) on the age scale
python scripts/predict_ages.py --targets 2 --mix_in_years --model_lv1 ridge --model_lv2 ridge

# Cross-decomposition route: Lv1 outputs 20 PLS components per modality -> XGBoost (Lv2)
python scripts/predict_ages.py --lv1_output cross-decomp --pls_components 20 --model_lv2 xgboost

# Cross-decomposition with OOF permutation Parallel Analysis selecting K per modality
python scripts/predict_ages.py --lv1_output cross-decomp --pls_components auto --pls_n_perm 100 --model_lv2 ridge

# Train only Level-1 models for specific feature blocks (useful for parallel runs across servers)
python scripts/predict_ages.py --only_run_lv1_models --feat_types DTI_FA DTI_MD

```

### 3. Feature-Block Importance & Ablation (`eval_feats_imp.py`)

Automatically infers `targets`, `mix_in_years`, `ALPHAS_LOWER` (for z-scored `Mix`), and `feat_src` (`targ-preds` vs `cross-decomp`) from the selected Level-2 run directory:

```
# Evaluate the latest Level-2 run with drop-one and only-one ablation across 5 seeds
python scripts/eval_feats_imp.py --n_seeds 5

# Target a specific run and include forward/backward stepwise selection
python scripts/eval_feats_imp.py --lv2_dir results/<lv1_key>/<lv2_key> --modes only_one drop_one forward backward

```

### 4. Fit & Coefficient Summaries (`summ_results.py`)

Extracts per-split regression fits (`pred_fits.csv`) and Level-2 model weights or tree feature importances (`coefficients.csv`):

```
python scripts/summ_results.py
# Or specify a run explicitly:
python scripts/summ_results.py -p results/<lv1_key>/<lv2_key>/predictions.csv

```

### 5. PAD × Cognitive Score Correlations (`calc_corr.py`)

Computes partial or raw correlations between PAD and individual cognitive scores across refitting seeds, aggregates via Fisher's $z$, and applies Benjamini–Hochberg FDR correction:

```
# Partial correlation controlling for chronological age (default -pad 1), Pearson (-m 0)
python scripts/calc_corr.py -d results/<lv1_key>/<lv2_key>/eval_<timestamp> -pad 1 -m 0 -fdr 0.05

```

### 6. Visualization (`plot_results.py`)

Generates scatter plots, Level-2 block weight/importance bars (signed sum for linear predictions, $L_2$ norm for linear PLS components, direct sum for tree importances), MAE bars, correlation boxplots, and correlation heatmaps:

```
# Draw all figures for the latest evaluation run
python scripts/plot_results.py --to_draw 0

# Draw specific figures only (1: age_scatters, 2: coef_bars, 3: mae_bars, 4: corr_boxes, 5: corr_heat)
python scripts/plot_results.py --to_draw 2 3 4 -d results/<lv1_key>/<lv2_key>/eval_<timestamp>

```

## Output Directory Layout

Results, saved pipelines, and figures share a mirrored hierarchy:

```
results/<target_key>_<mdl_tag>_cv<n_folds>_<seed>/
├── <modality>.csv                                  # Level-1 per-fold & test metrics
└── <model_lv2>_<seed>_<YYYY-MM-DD_HH-MM-SS>/
    ├── predictions.csv                             # Merged Lv1 + Lv2 (raw & age-corrected) predictions
    ├── pls_feats.csv                               # Saved under --lv1_output cross-decomp
    ├── performance.csv                             # Level-2 CV & test metrics
    ├── age-correction_params.csv                   # Per-fold Cole age-bias calibration parameters
    ├── pred_fits.csv & coefficients.csv            # Generated by summ_results.py
    ├── summary.json & logs.txt                     # Full run metadata & execution log
    └── eval_<YYYY-MM-DD_HH-MM-SS>/
        ├── performance_organized.csv               # Ranked ablation / stepwise summary
        ├── performance_listed.csv                  # Per-seed evaluation records
        ├── predictions_seed-<seed>.csv             # Per-seed subset predictions
        └── pad-<type>_x_<score> (<method>)/        # Generated by calc_corr.py

```

* `<target_key>`: `1y` (`Age`), `2y` (`Age + Cog`), `mix` (z-scored `Mix`), or `mixa` (`Mix` in years).

* `<mdl_tag>`: Level-1 model name, with `-comp<K>` / `-comp-auto` / `-comp-cv` under `cross-decomp`, or `-k<K>` / `-auto` for non-CV PLS under `targ-preds`.