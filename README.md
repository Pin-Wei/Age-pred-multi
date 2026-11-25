# Multimodal Brain Age Prediction

A two-level multimodal machine learning pipeline for predicting chronological age (`Age`), cognitive performance (`Cog`), or an equal-weight composite of both (`Mix`) from neuroimaging, electrophysiological, and behavioral features, followed by feature-block ablation, predicted age difference (PAD, predicted age − chronological age) correlation analysis, and publication-ready visualization.

## 1. Overview

The pipeline integrates multiple brain and behavioral modalities through a two-level stacking architecture:

- **Level 1 (Modality-Specific Models)**: Fits per-modality cross-validated models independently on each feature block (`DTI_FA`, `DTI_MD`, `rs-EEG_PSD`, `rs-MRI_FC`, `MRI_ROI`, `FUN_COGNITIVE`, `BEH_COGNITIVE`) and incorporates brain-age predictions from the pre-trained [Pyment](https://github.com/estenhl/pyment-public) 3-D CNN model. Level 1 supports two output routes (`--lv1_output`):

   * **`targ-preds`** *(default)*: Passes out-of-fold / test target predictions from each modality to Level 2.

   * **`cross-decomp`**: Projects each modality onto supervised PLS latent components (with fixed *K*, inner-CV selection, or cross-validated permutation Parallel Analysis `--pls_components auto`, sign-aligned across folds to fold 0).

- **Level 2 (Multimodal Stacking Model)**: Trains a cross-validated second-level regressor (meta-model, `--model_lv2`, with median imputation for missing modalities) on the merged Level-1 predictions or PLS components to produce the final multimodal predictions, applies Cole's linear age-bias correction per fold, and (for `Mix` targets) maps composite predictions back to `Age` and `Cog` scales.

- **Downstream Evaluation**: Summarizes regression fits and Level-2 weights/importances (`summ_results.py`), evaluates modality importance via multi-seed ablation and stepwise selection (`eval_feats_imp.py`), computes Benjamini–Hochberg FDR-corrected PAD–cognition correlations (`calc_corr.py`), and generates publication-ready figures (`plot_results.py`).

## 2. Repository Structure

```
Age-pred-multi/
├── etc/
│   ├── pyment_config.json
│   ├── pyment_preprocess.Dockerfile
│   └── pyment_preprocess_and_finetune.Dockerfile
├── scripts/
│   ├── fetch_dropbox_files.py                    # Download resting-state EEG/fMRI recordings (credentials in scripts/.env)
│   ├── run_tbss.sh                               # FSL TBSS on FA maps, then projection onto MD maps
│   ├── prep_pyment.py, run_pyment.sh             # Prepare T1w inputs and run Pyment container (1: fine-tune, 2: predict)
│   ├── make_df_preproc.py                        # Merge/clean source tables into `df_preproc.csv` & `preproc_features.txt`
│   ├── make_df_scores.py                         # Build `df_scores_ST.csv` & composite cognitive score (cog_scores.json)
│   ├── make_df_dti_roi-means.py                  # Extract DTI ROI mean features (.csv and .npz)
│   ├── make_df_rs-eeg_psd.py                     # Extract resting-state EEG PSD features (MNE)
│   ├── make_df_rs-fmri_fc.py                     # Extract resting-state fMRI FC features (Nilearn)
│   ├── models.py                                 # Estimator classes: TargetScaler, MSVR, MOGP (ICM)
│   ├── training.py                               # Function `train_eval_model` and its parts: pipeline construction, hyperparameter search, PLS component selection, metrics, age-bias correction
│   ├── helpers.py                                # Functions for data loading (TBSS cache, feature tables) & latest-run lookup
│   ├── plotting.py                               # Plotting functions & the shared `Style`
│   ├── utils.py                                  # Functions for logging (custom_print, tee_output, mute_print), JSON conversion
│   ├── predict_ages.py                           # Main two-level training & prediction pipeline
│   ├── summ_results.py                           # Fit predicted on chronological age (pred_fits.csv) & export Level-2 weights (coefficients.csv)
│   ├── eval_feats_imp.py                         # Multi-seed modality ablation & stepwise evaluation
│   ├── calc_corr.py                              # Calculate PAD x cognitive score (partial) correlations with BH-FDR
│   └── plot_results.py                           # Generate scatter, weight/importance, MAE, box & heatmap plots
├── environment.yml                               # Pinned Conda environment (conda-forge, Python 3.12)
├── environment.txt                               # Exact linux-64 Conda @EXPLICIT lockfile
└── .gitignore
```

Runtime directories (`data/`, `models/`, `results/`, and `figures/`) are untracked by git. All scripts resolve paths from their own location (`scripts/..` is the project root), so they can be run from any working directory.

## 3. Installation

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

* `py-xgboost` is installed with the CUDA build and trains on GPU or CPU according to `--xgb_device`, automatically falling back to CPU when no CUDA GPU is available or GPU memory is exhausted.

* `run_tbss.sh` requires FSL (`$FSLDIR`), and `run_pyment.sh` requires Docker with GPU support (Dockerfiles and config are in `etc/`).

## 4. Supported Modalities, Targets & Models

### 4.1. Feature Blocks

| Block | Description | Source |
| ----- | ----- | ----- |
| `DTI_FA`, `DTI_MD` | 4-D TBSS voxel-wise FA / MD within the mean-FA mask, downsampled by `--downsample` (default `2`) and cached as `.npy` | `run_tbss.sh` |
| `rs-MRI_FC` | Resting-state fMRI functional connectivity | `make_df_rs-fmri_fc.py` |
| `FUN_COGNITIVE` | Task-evoked MRI / EEG cognitive features (`LANGUAGE`, `MEMORY`, `MOTOR`) | `make_df_preproc.py` |
| `MRI_ROI` | Regional structural T1w MRI measures (`STRUCTURE*`, excluding FA) | `make_df_preproc.py` |
| `BEH_COGNITIVE` | Behavioral cognitive features (`LANGUAGE`, `MEMORY`, `MOTOR`) | `make_df_preproc.py` |
| `rs-EEG_PSD` | Resting-state EEG power spectral density | `make_df_rs-eeg_psd.py` |
| `Pyment` | Pre-trained 3-D CNN brain-age predictions from T1w MRI; enters Level 2 directly as one column | `run_pyment.sh` |

Participants missing data for a block are omitted from that Level-1 model, and the resulting gaps are median-imputed at Level 2. `--feat_types` restricts Level 1 to specified blocks (`Pyment` is always added at Level 2).

### 4.2. Targets (`--targets`)

| Value | Target(s) | Description | 
| ----- | ----- | ----- | 
| `0` | `Age` | Single-output chronological age prediction | 
| `1` *(default)* | `Age`, `Cog` | Multi-output prediction; `Cog` is the mean of three z-scored domain subtests (`make_df_scores.py`), rescaled to `Age`'s SD during fitting via `TargetScaler` | 
| `2` | `Mix` = `[ z(Age) ± z(Cog) ] / 2` | Equal-weight composite standardized on training rows; `Cog` takes the sign of its training correlation with `Age`. With `--mix_in_years`, scaled to `mean(Age) + SD(Age) × Mix` | 

Under `--targets 2`, every `Mix` prediction is mapped back to `Age` and `Cog` via training-set least-squares fits (`restore_from_mix`), so downstream scripts always find `Age_*` and `Cog_*` columns (the `Age` fit is the identity under `--mix_in_years`).

### 4.3. Regression Algorithms (`--model_lv1` / `--model_lv2`)

| Key | Model | Hyperparameter Tuning & Scaling | 
| ----- | ----- | ----- | 
| `elasticnet` | `ElasticNetCV` / `MultiTaskElasticNetCV` | Inner 5-fold CV over `--l1_ratios` × `--alphas` | 
| `lasso` | `LassoCV` / `MultiTaskLassoCV` | Inner 5-fold CV over `--alphas` | 
| `ridge` | `RidgeCV` | Inner 5-fold CV over `--alphas` | 
| `xgboost` | `XGBRegressor` | `--opt_trials` Optuna TPE trials (reused from cached models when available); no feature scaling | 
| `msvr` | Custom RBF Multi-Output SVR (joint-intercept IRWLS) | Inner 5-fold `GridSearchCV` over `C`, `epsilon`, `gamma`; targets scaled with `StandardScaler` | 
| `mogp` | Custom GPyTorch ICM Multi-Output GP | Exact Marginal Log-Likelihood (Adam); targets scaled with `StandardScaler` | 
| `pls` | `PLSRegression` | Fixed `K`, inner 5-fold CV (`cv`), or OOF Parallel Analysis (`auto`) | 

Features are standardized within each training fold (except for `xgboost`), and inner CV uses 5 folds with `--seed_inner`.

## 5. Usage

### 5.1. Data Preparation (one-time)

Download raw recordings and build tabular features, cognitive scores, TBSS volumes, and Pyment predictions:

```
python scripts/fetch_dropbox_files.py
bash   scripts/run_tbss.sh                  # stages: all | 1 | 2 | 3 | 4 | md
python scripts/make_df_preproc.py
python scripts/prep_pyment.py && bash scripts/run_pyment.sh 2
python scripts/make_df_rs-eeg_psd.py
python scripts/make_df_rs-fmri_fc.py
python scripts/make_df_scores.py            # required whenever Cog or Mix is a target
```

### 5.2. Two-Level Training & Prediction (`predict_ages.py`)

Splits the training set into `--n_folds` outer folds (`--seed`): training predictions are out-of-fold, test predictions come from the fold with the lowest validation MAE, and Level 2 is trained on these out-of-fold outputs with per-fold Cole age-bias correction (`C-*`). Existing `.joblib` models are reused unless `--overwrite_mdl` or `--add_new_mdls` is passed.

Under `--lv1_output cross-decomp` (requires `--n_folds > 1`), `--pls_components` selects *K* once on the training split: an integer *K* *(default `20`)*, `cv` (lowest inner-CV MAE in `1..--pls_max_components`), or `auto` (cross-validated permutation Parallel Analysis selecting leading components whose out-of-fold $\Delta R^2 > 0$ exceeds the 95th percentile of `--pls_n_perm` target permutations). Each fold's PLS components are sign-aligned to fold 0, with cross-fold weight `|cos|` logged per component; `Pyment` still enters Level 2 as a single predicted-age column.

```
# Default: Age + Cog, ElasticNet (Lv1) -> Ridge (Lv2)
python scripts/predict_ages.py

# Age only, XGBoost at Level 2 on CPU
python scripts/predict_ages.py --targets 0 --model_lv2 xgboost --xgb_device cpu

# Equal-weight composite target (Mix) on the age scale
python scripts/predict_ages.py --targets 2 --mix_in_years --max_iter 50000

# Cross-decomposition route: Lv1 outputs K = 5 PLS components per modality -> Ridge (Lv2)
python scripts/predict_ages.py --lv1_output cross-decomp --pls_components 5

# Cross-decomposition with OOF permutation Parallel Analysis selecting K per modality
python scripts/predict_ages.py --lv1_output cross-decomp --pls_components auto --pls_n_perm 100

# Train only Level-1 models for specific feature blocks (useful for parallel runs across servers)
python scripts/predict_ages.py --only_run_lv1_models --feat_types DTI_FA DTI_MD
```

### 5.3. Fit & Coefficient Summaries (`summ_results.py`)

Computes per-split (`all`, `test`, `train`) linear regression fits, correlations, MAE, and $R^2$ (`pred_fits.csv`), and extracts Level-2 linear coefficients or tree feature importances across folds (`coefficients.csv`):

```
# Summarize the latest Level-2 run
python scripts/summ_results.py

# Or target a specific run explicitly
python scripts/summ_results.py -p results/<lv1_key>/<lv2_key>/predictions.csv
```

### 5.4. Feature-Block Importance & Ablation (`eval_feats_imp.py`)

Groups Level-2 input columns into modality blocks (plus `Pyment`) and refits Level 2 on subsets (`--modes`: `only_one`, `drop_one`, `forward`, `backward`, ranked by `--score_by`, default `Test_R2`) across `--n_seeds` (default `5`) reproducible seeds. Automatically infers `targets`, `mix_in_years`, `ALPHAS_LOWER` (for z-scored `Mix`), and `feat_src` (`targ-preds` vs `cross-decomp`) from the selected run folder name, and reuses saved Level-2 XGBoost hyperparameters. Note that custom `--alphas` grids are not read back from saved runs and should be passed explicitly if used.

```
# Evaluate the latest Level-2 run with only_one and drop_one ablation across 5 seeds
python scripts/eval_feats_imp.py --n_seeds 5

# Target a specific run and include forward/backward stepwise selection
python scripts/eval_feats_imp.py --lv2_dir results/<lv1_key>/<lv2_key> --modes only_one drop_one forward backward
```

### 5.5. PAD × Cognitive Score Correlations (`calc_corr.py`)

Correlates PAD from every model in `predictions_seed-*.csv` with every score in `df_scores_ST.csv` that was not used to build `Cog`. Supports `-pad` (`0` age-corrected `C-*`, `1` raw with age partialled out of PAD *\[default\]*, `2` raw), `-ps` (also partial age out of the scores), `-m` (`0` Pearson, `1` Spearman), and `-set` (`0` all, `1` test, `2` train). Averages correlations across seeds via Fisher's *z*, applies Benjamini–Hochberg FDR (`-fdr`, default `0.05`; `-fs 0` across all models or `-fs 1` per model), and tests per-model displacement from zero via Wilcoxon signed-rank tests:

```
# Partial Pearson correlation controlling for chronological age (defaults)
python scripts/calc_corr.py

# Target a specific evaluation folder explicitly
python scripts/calc_corr.py -d results/<lv1_key>/<lv2_key>/eval_<timestamp> -pad 1 -m 0 -fdr 0.05
```

### 5.6. Visualization (`plot_results.py`)

Generates scatter plots, Level-2 block weight/importance bars, ablation MAE bars, correlation boxplots, and correlation heatmaps (`--fig_scale`, `--font_scale`, `--dpi`). Because `plot_results.py` locates the Level-2 run via `eval_<timestamp>/`, `--to_draw 1 2` requires `summ_results.py` and `eval_feats_imp.py` to have run, while `--to_draw 3 4 5` also requires `calc_corr.py`:

```
# Draw all figures for the latest evaluation run
python scripts/plot_results.py --to_draw 0

# Draw specific figures for a pinned run
python scripts/plot_results.py --to_draw 2 3 4 -d results/<lv1_key>/<lv2_key>/eval_<timestamp>
```

| ID | Output File | Content | 
| ----- | ----- | ----- | 
| `1` | `[scatter] Fits between real and predicted ages (<model>).png` | Predicted vs. chronological age per Level-1 model, `Pyment`, and Level 2 | 
| `2` | `[bars] Coefficients of the fold-<k> 2nd-level model (<target>).png` | Linear Level 2 on `targ-preds`: signed sum of weights per block | 
| `2` | `[bars] Weight norms (L2) of the fold-<k> 2nd-level model (<target>).png` | Linear Level 2 on `cross-decomp`: $L_2$ norm of weights per block (sign-invariant and fair against 1-column `Pyment`) | 
| `2` | `[bars] Importances of the fold-<k> 2nd-level model (Importance).png` | XGBoost Level 2: gain importances summed per block (bars sum to 1) | 
| `3` | `[bars] MAE of 2nd-level models (<set>).png` | Ablation model MAE (mean ± SD across seeds) with full-model baseline | 
| `4` | `[boxes] Correlations between PAD[-<type>] and ST scores (<set>).png` | Distribution of each model's PAD–score correlations with FDR significance | 
| `5` | `[heat] Correlations between PAD[-<type>] and ST scores (<set>).png` | Model × score correlation heatmap grouped by cognitive domain | 

**Tip for concurrent runs**: When multiple runs are active at once, pass paths explicitly across the chain rather than relying on the latest-run lookup:

```
LV2=results/2y_pls-comp5_cv5_42/ridge_42_<timestamp>
python scripts/summ_results.py -p $LV2/predictions.csv
python scripts/eval_feats_imp.py --lv2_dir $LV2
python scripts/calc_corr.py -d $LV2/eval_<timestamp>
python scripts/plot_results.py -d $LV2/eval_<timestamp>
```

## 6. Output Directory Layout

Results, saved pipelines (`models/`), and figures (`figures/`) share a mirrored hierarchy:

```
results/<lv1_key>/
├── <modality>.csv                                  # Level-1 per-fold, pooled, and test MAE / R²
└── <lv2_key>/
    ├── predictions.csv                             # Per participant: targets, Lv1 outputs, *_Final, C-*_Final
    ├── pls_feats.csv                               # PLS components handed to Level 2 (cross-decomp only)
    ├── performance.csv                             # Level-2 CV & test MAE / R²
    ├── age-correction_params.csv                   # Per-fold Cole age-bias calibration (intercept & slope)
    ├── pred_fits.csv & coefficients.csv            # Generated by summ_results.py
    ├── summary.json & logs.txt                     # Full run metadata & execution log
    └── eval_<YYYY-MM-DD_HH-MM-SS>/
        ├── performance_organized.csv               # Ranked ablation / stepwise summary
        ├── performance_listed.csv                  # Per-seed evaluation records
        ├── predictions_seed-<seed>.csv             # Per-seed subset predictions
        └── pad-<type>_x_<score> (<method>)/        # Generated by calc_corr.py (long, mae, by-score, by-model)
```

| Key | Pattern | Example | 
| ----- | ----- | ----- | 
| `lv1_key` | `{target_key}_{mdl_tag}_cv{n_folds}_{seed}` | `2y_elasticnet_cv5_42` | 
|  | Under `--lv1_output cross-decomp`: `mdl_tag` is `pls-comp{K}`, `pls-comp-auto`, or `pls-comp-cv` | `2y_pls-comp5_cv5_42` | 
|  | Under `--model_lv1 pls` with non-`cv` `--pls_components`: `mdl_tag` is `pls-k{K}` or `pls-auto` | `2y_pls-k3_cv5_42` | 
| `lv2_key` | `{model_lv2}_{seed}_{YYYY-MM-DD_HH-MM-SS}` | `ridge_42_2026-11-24_18-42-58` | 

* `{target_key}` is `1y` (`Age`), `2y` (`Age + Cog`), `mix` (z-scored `Mix`), or `mixa` (`Mix` in years).

* `mdl_tag` never contains underscores (`_`), as downstream scripts split `lv1_key` on `_`. `--add_new_mdls` appends `+` until an unused folder name is found.

## 7. Notes on Interpretation

* **Composite-target PAD circularity**: Under `--targets 2`, PAD contains `SD(Age) × [ ±z(Cog) − z(Age) ] / 2` by construction (i.e. `−SD(Age) × [ z(Age) + z(Cog) ] / 2` when `Age` and `Cog` are negatively correlated), regardless of whether `--mix_in_years` is set. A correlation between this PAD and cognitive scores is therefore not by itself evidence that brain features carry cognitive signal.

* **Regularization scale for `Mix`**: Without `--mix_in_years`, `Mix` has SD ≈ 1, so `--alphas` automatically defaults to `ALPHAS_LOWER` (`1e-3` to `1e3`) rather than `ALPHAS` (`0.1` to `1e5`).

* **Comparing ablation runs**: `eval_feats_imp.py` redraws outer CV folds for each seed on fixed Level-1 outputs from the original run. Compare ablation models and `N_sig` within a single `eval_<timestamp>/` folder (or across folders produced with the same seeds), rather than directly against the original Level-2 run's absolute accuracy.

* **Legacy `cross-decomp` runs**: For runs predating per-run `pls_feats.csv`, `eval_feats_imp.py` falls back to `results/<lv1_key>/pls_feats.csv` and then `data/tabular/df_pls-feats.csv`; check the `From:` line in the log to confirm the source file.

* **PLS sign reference**: Under `cross-decomp`, fold-0 weights serve as the sign-alignment reference for all other folds, so Level-2 results depend on this reference.
