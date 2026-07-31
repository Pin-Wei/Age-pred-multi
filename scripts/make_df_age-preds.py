#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from utils import print_missing, get_tbss_processed, load_feat_table, train_eval_model


class Config:
    def __init__(self):
        self.setup_model_params()
        self.setup_vars()
        self.setup_paths()

    def setup_model_params(self):
        self.model_type = ["elasticnet", "ridge"][0]
        self.l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
        self.alphas = [1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5]
        self.n_folds = 5
        self.seed = 42
        self.seed_inner = 0
        self.max_iter = 10000
        self.n_jobs = -1
        self.verbose = 1
        self.overwrite_mdl = bool(0) 
        
    def setup_vars(self):
        self.downsmple = 2
        self.skeleton = 1

    def setup_paths(self):
        self.proj_root       = Path(__file__).resolve().parents[1]
        self.tbl_dir         = self.proj_root / "data" / "tabular"
        self.df_preproc_path = self.tbl_dir / "df_preproc.csv"
        self.tbl_out_path    = self.tbl_dir / f"df_{self.model_type}_age-preds.csv"
        self._setup_paths_for_tbss()
        self._setup_paths_for_others()

    def _setup_paths_for_tbss(self):
        tbss_dir  = self.proj_root / "data" / "dti" / "tbss"
        self.fa_3d_dir          = tbss_dir / "origdata"
        self.fa_3d_paths        = list(self.fa_3d_dir.glob("*.nii.gz"))
        self.fa_4d_path         = tbss_dir / "stats" / f"all_FA{['', '_skeletonised'][self.skeleton]}.nii.gz"
        self.md_4d_path         = tbss_dir / "stats" / f"all_MD{['', '_skeletonised'][self.skeleton]}.nii.gz"
        self.fa_mask_path       = tbss_dir / "stats" / f"mean_FA{['', '_skeleton'][self.skeleton]}_mask.nii.gz"
        
        key = f"{['', '-sk'][self.skeleton]}_ds{self.downsmple}"
        self.fa_npy_path        = self.proj_root / "data" / "dti" / f"tbss_FA{key}_flat.npy"
        self.md_npy_path        = self.proj_root / "data" / "dti" / f"tbss_MD{key}_flat.npy"
        
        dti_model_dir = self.proj_root / "models" / f"dti_{self.model_type}"
        self.fa_model_path      = str(dti_model_dir) + "/" + f"FA{key}_{{}}.joblib"
        self.md_model_path      = str(dti_model_dir) + "/" + f"MD{key}_{{}}.joblib"
        # self.fm_model_path      = str(dti_model_dir) + "/" + f"FA+MD{key}_{{}}.joblib"
        self.fa_model_perf_path = dti_model_dir / f"FA{key}_performance.csv"
        self.md_model_perf_path = dti_model_dir / f"MD{key}_performance.csv"
        # self.fm_model_perf_path = dti_model_dir / f"FA+MD{key}_performance.csv"

    def _setup_paths_for_others(self):
        model_dir = self.proj_root / "models"

        self.dti_rois_df_path      = self.tbl_dir / "df_dti_roi-means.csv"
        self.dti_rois_df_mdl_path  = str(model_dir) + "/" + f"dti_roi_{self.model_type}_{{}}.joblib"
        self.dti_rois_df_perf_path = model_dir / f"dti_roi_{self.model_type}_performance.csv"

        self.rseeg_psd_df_path   = self.tbl_dir / "df_rs-eeg_psd.csv"
        self.rseeg_psd_mdl_path  = str(model_dir) + "/" + f"rs-eeg_psd_{self.model_type}_{{}}.joblib"
        self.rseeg_psd_perf_path = model_dir / f"rs-eeg_psd_{self.model_type}_performance.csv"

        self.rsfmri_fc_df_path      = self.tbl_dir / "df_rs-fmri_fc.csv"
        self.rsfmri_fc_df_mdl_path  = str(model_dir) + "/" + f"rs-fmri_fc_{self.model_type}_{{}}.joblib"
        self.rsfmri_fc_df_perf_path = model_dir / f"rs-fmri_fc_{self.model_type}_performance.csv"


def load_data(feat_name, tbl_path, config):
    if feat_name in ["DTI_FA", "DTI_MD"]:
        X_subjs = [ fp.name.split(".")[0] for fp in sorted(config.fa_3d_paths) ]  # the order TBSS merge per-subject FA volumes

    if feat_name == "DTI_FA":
        X = get_tbss_processed(
            img_path=config.fa_4d_path, 
            N=len(X_subjs), 
            mask_path=config.fa_mask_path, 
            stride=config.downsmple, 
            cache=config.fa_npy_path
        )
    elif feat_name == "DTI_MD":
        X = get_tbss_processed(
            img_path=config.md_4d_path, 
            N=len(X_subjs), 
            mask_path=config.fa_mask_path, 
            stride=config.downsmple, 
            cache=config.md_npy_path
        )
    else:
        X_subjs, X = load_feat_table(tbl_path)
    
    return X_subjs, X


def run_age_preds(subj_df: pd.DataFrame, config: Config) -> list[pd.DataFrame]:
    age_pred_dfs = []

    for feat_name, tbl_path, model_path, perf_path in [
        ("DTI_FA",     None,                     config.fa_model_path,         config.fa_model_perf_path), 
        ("DTI_MD",     None,                     config.md_model_path,         config.md_model_perf_path), 
        ("DTI_ROI",    config.dti_rois_df_path,  config.dti_rois_df_mdl_path,  config.dti_rois_df_perf_path), 
        ("rs-MRI_FC",  config.rsfmri_fc_df_path, config.rsfmri_fc_df_mdl_path, config.rsfmri_fc_df_perf_path), 
        ("rs-EEG_PSD", config.rseeg_psd_df_path, config.rseeg_psd_mdl_path,    config.rseeg_psd_perf_path), 
    ]:
        print(f"\nLoading {feat_name} data ...")
        X_subjs, X = load_data(feat_name, tbl_path, config)
        print_missing(subj_df.index.tolist(), X_subjs)

        ages = np.array([ subj_df["BASIC_INFO_AGE"][s] for s in X_subjs ], dtype=np.float32)
        sets = np.array([ subj_df["Set"][s] for s in X_subjs ])
        idx_tr = np.where(sets == "train")[0]
        idx_te = np.where(sets == "test")[0]

        print(f"\nTrain and eval {config.model_type} models on {feat_name} features ...")
        y_pred, fold_n = train_eval_model(
            X, ages, idx_tr, idx_te, 
            model_type=config.model_type, 
            seed=config.seed, 
            seed_inner=config.seed_inner, 
            n_folds=config.n_folds, 
            l1_ratios=config.l1_ratios, 
            alphas=config.alphas, 
            max_iter=config.max_iter, 
            n_jobs=config.n_jobs, 
            verbose=config.verbose, 
            model_path_template=model_path, 
            model_perf_path=perf_path, 
            overwrite=config.overwrite_mdl
        )

        age_pred_dfs.append(
            pd.DataFrame({    
                "SID": X_subjs, 
                f"Fold_{feat_name}": fold_n, 
                f"Age_{feat_name}": y_pred, 
            })
        )

    return age_pred_dfs


def main():
    config = Config()

    subj_df = pd.read_csv(config.df_preproc_path, index_col=["BASIC_INFO_ID"], usecols=["BASIC_INFO_ID", "BASIC_INFO_AGE", "Set"])
    n_subjs = len(subj_df)
    print(f"\nNumber of participants: {n_subjs}")

    age_pred_dfs = run_age_preds(subj_df, config)

    out_df = (
        subj_df
        .reset_index()
        .rename(columns={
            subj_df.index.name: "SID", 
            "BASIC_INFO_AGE": "Age"
        })
    )
    for df in age_pred_dfs:
        out_df = pd.merge(out_df, df, on="SID", how="left")

    out_df.to_csv(config.tbl_out_path, index=False)
    print(f"\nSaved: {config.tbl_out_path}\n")


if __name__ == "__main__":
    main()
    