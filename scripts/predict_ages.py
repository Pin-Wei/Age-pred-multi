#!/usr/bin/env python3


import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from utils import print_missing, get_tbss_processed, load_feat_table, train_eval_model, to_json_compatible


MODELS    = ["elasticnet", "ridge"]
L1_RATIOS = [.1, .5, .7, .9, .95, .99, 1]
ALPHAS    = [1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5]


class Config:
    def __init__(self, args: argparse.Namespace = None):
        self.args = args if args is not None else parse_args([])
        self.setup_model_params()
        self.setup_vars_and_paths()

    def setup_model_params(self):
        self.model_lv1 = self.args.model_lv1
        self.model_lv2 = self.args.model_lv2
        self.l1_ratios = list(self.args.l1_ratios)
        self.alphas = list(self.args.alphas)
        self.n_folds = self.args.n_folds
        self.seed = self.args.seed
        self.seed_inner = self.args.seed_inner
        self.max_iter = self.args.max_iter
        self.n_jobs = self.args.n_jobs
        self.verbose = self.args.verbose
        self.overwrite_mdl = self.args.overwrite_mdl

    def setup_vars_and_paths(self):
        self.proj_root = Path(__file__).resolve().parents[1]
        self.tbl_dir = self.proj_root / "data" / "tabular"
        self.pyment_tbl_path = self.proj_root / "data" / "pyment" / "predictions" / "predictions.csv"
        
        lv1_key = f"{self.model_lv1}_cv{self.n_folds}_{self.seed}"
        self.lv1_mdl_dir = self.proj_root / "models" / lv1_key
        self.lv1_res_dir = self.proj_root / "results" / lv1_key
        
        lv2_key = f"{self.model_lv2}_{self.seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.lv2_mdl_dir = self.lv1_mdl_dir / lv2_key
        self.lv2_res_dir = self.lv1_res_dir / lv2_key
        self.pred_out_path = self.lv2_res_dir / "all_preds.csv"
        self.summ_out_path = self.lv2_res_dir / "summary.json"
        self.log_out_path  = self.lv2_res_dir / "logs.txt"

        preproc_f_names = self._for_preproc()
        tbss_key = self._for_tbss()
        self._for_models(preproc_f_names, tbss_key)

    def _for_preproc(self):
        self.df_preproc_path = self.tbl_dir / "df_preproc.csv"
        preproc_feats_path = self.proj_root / "data" / "meta" / "preproc_features.txt"

        feats = pd.read_csv(preproc_feats_path, header=None).squeeze().tolist()
        self.sel_feats_by_name = {
            "MRI_ROI": [ f for f in feats if f.startswith("STRUCTURE") and not f.endswith("FA") ]
        }
        # for approach, domain in product(["MRI", "EEG", "BEH"], ["LANGUAGE", "MEMORY", "MOTOR"]):
        #     f_name = f"{approach}_{domain}"
        #     sel_feats = [ f for f in feats if f.startswith(domain) and approach in f ]
        #     if sel_feats:
        #         self.sel_feats_by_name.update({f_name: sel_feats})

        self.sel_feats_by_name.update({
            "FUN_COGNITIVE": [ 
                f for f in feats if any( f.startswith(dom) for dom in ["LANGUAGE", "MEMORY", "MOTOR"] ) 
                and any( app in f for app in ["MRI", "EEG"] ) 
            ], 
            "BEH_COGNITIVE": [
                f for f in feats if any( f.startswith(dom) for dom in ["LANGUAGE", "MEMORY", "MOTOR"] ) 
                and "BEH" in f
            ]
        })

        return list(self.sel_feats_by_name.keys())

    def _for_tbss(self):
        self.downsmple = self.args.downsample
        skeleton = int(self.args.skeleton)

        tbss_dir          = self.proj_root / "data" / "dti" / "tbss"
        self.fa_3d_dir    = tbss_dir / "origdata"
        self.fa_3d_paths  = list(self.fa_3d_dir.glob("*.nii.gz"))
        self.fa_4d_path   = tbss_dir / "stats" / f"all_FA{['', '_skeletonised'][skeleton]}.nii.gz"
        self.md_4d_path   = tbss_dir / "stats" / f"all_MD{['', '_skeletonised'][skeleton]}.nii.gz"
        self.fa_mask_path = tbss_dir / "stats" / f"mean_FA{['', '_skeleton'][skeleton]}_mask.nii.gz"
        
        key = f"{['', '-sk'][skeleton]}_ds{self.downsmple}"
        self.fa_npy_path = self.proj_root / "data" / "dti" / f"tbss_FA{key}_flat.npy"
        self.md_npy_path = self.proj_root / "data" / "dti" / f"tbss_MD{key}_flat.npy"
        
        return key

    def _for_models(self, preproc_f_names, tbss_key):
        feat_name_2_affix = {
            "DTI_FA"    : f"dti_fa{tbss_key}", 
            "DTI_MD"    : f"dti_md{tbss_key}", 
            "DTI_ROI"   : "dti_roi-means", 
            "rs-EEG_PSD": "rs-eeg_psd", 
            "rs-MRI_FC" : "rs-fmri_fc", 
            "MRI_ROI"   : "t1w_roi-means"
        }
        feat_name_2_affix.update({
            f_name: f_name.lower() 
            for f_name in preproc_f_names
            if f_name != "MRI_ROI"
        })
        self.feat_types  = list(feat_name_2_affix.keys())
        if self.args.feat_types:  # the paths below stay complete; only the first level's loop shrinks
            unknown = [ f for f in self.args.feat_types if f not in feat_name_2_affix ]
            assert not unknown, f"\nUnknown --feat-types {unknown}; available: {self.feat_types}\n"
            self.feat_types = [ f for f in self.feat_types if f in set(self.args.feat_types) ]

        self.tbl_paths   = {}
        self.model_paths = {}
        self.perf_paths  = {}

        for f_name, affix in feat_name_2_affix.items():
            if f_name in ["DTI_FA", "DTI_MD"]:
                self.tbl_paths[f_name] = None
            elif f_name in preproc_f_names:
                self.tbl_paths[f_name] = self.df_preproc_path
            else:
                self.tbl_paths[f_name] = self.tbl_dir / f"df_{affix}.csv" 
            self.model_paths[f_name]   = self.lv1_mdl_dir / f"{affix}_{{}}.joblib"
            self.perf_paths[f_name]    = self.lv1_res_dir / f"{affix}.csv"

        self.lv2_model_path = self.lv2_mdl_dir / f"pipeline_{{}}.joblib"
        self.lv2_perf_path  = self.lv2_res_dir / f"performance.csv"


@contextmanager
def tee_output(log_path: Path):
    '''
    Duplicate stdout/stderr to both the terminal and `log_path`.

    Redirection happens at the file-descriptor level, 
    so it also captures output from C extensions and joblib/loky workers, not just `print`.
    '''
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()  # clears the internal memory 
    sys.stderr.flush()

    proc = subprocess.Popen(["tee", str(log_path)], stdin=subprocess.PIPE)
    saved_fds = (os.dup(1), os.dup(2))  # file descriptors
    os.dup2(proc.stdin.fileno(), 1)
    os.dup2(proc.stdin.fileno(), 2)
    was_line_buffered = sys.stdout.line_buffering
    sys.stdout.reconfigure(line_buffering=True)  # keep ordering vs. stderr / workers
    
    try:
        yield

    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.reconfigure(line_buffering=was_line_buffered)
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        for fd in saved_fds:
            os.close(fd)
        proc.stdin.close()
        proc.wait()


def parse_args(argv: list[str] = None,
               parser: argparse.ArgumentParser = None,
               defaults: dict = None) -> argparse.Namespace:
    '''
    Command-line overrides for the Config values worth varying between runs.

    A downstream script sharing this Config passes its own `parser`, already holding its extra
    options, so that these options are added to it rather than to a fresh one. It may also pass
    `defaults` to replace the ones set below, which `argv` still overrides.
    '''
    parser = parser or argparse.ArgumentParser(
        description="Train and evaluate the two-level age-prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    grp_model = parser.add_argument_group("model")
    grp_model.add_argument("--model-lv1", choices=MODELS, default=MODELS[0],
                           help="model fitted per feature type; also names the first-level dirs")
    grp_model.add_argument("--model-lv2", choices=MODELS, default=MODELS[1],
                           help="model fitted on the first-level predicted ages")
    grp_model.add_argument("--l1-ratios", nargs="+", type=float, default=L1_RATIOS, metavar="R",
                           help="elasticnet mixing parameters to search over")
    grp_model.add_argument("--alphas", nargs="+", type=float, default=ALPHAS, metavar="A",
                           help="regularization strengths to search over")
    grp_model.add_argument("--n-folds", type=int, default=5,
                           help="outer CV folds of the first level; also names the first-level dirs")
    grp_model.add_argument("--max-iter", type=int, default=10000, help="solver iteration cap")
    grp_model.add_argument("--seed", type=int, default=42,
                           help="outer split seed; also names the first- and second-level dirs")
    grp_model.add_argument("--seed-inner", type=int, default=0, help="inner CV split seed")

    grp_data = parser.add_argument_group("data")
    grp_data.add_argument("--feat-types", nargs="+", default=None, metavar="FEAT",
                          help="restrict the first level to these feature types; None runs all of them")
    grp_data.add_argument("--downsample", type=int, default=2,
                          help="voxel stride for the TBSS volumes; also names their cached .npy files")
    grp_data.add_argument("--skeleton", action="store_true",
                          help="use the skeletonised TBSS volumes instead of the full ones")

    grp_run = parser.add_argument_group("runtime")
    grp_run.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs per fit")
    grp_run.add_argument("--verbose", type=int, default=1, help="verbosity of the fits")
    grp_run.add_argument("--overwrite-mdl", action="store_true",
                         help="refit and overwrite the models already saved on disk")

    parser.set_defaults(**(defaults or {}))  # after the options exist, so that --help shows the replacements

    return parser.parse_args(argv)


def load_data(f_name: str, config: Config) -> tuple[np.ndarray, np.ndarray | pd.DataFrame]:
    if f_name in ["DTI_FA", "DTI_MD"]:
        X_subjs = [ fp.name.split(".")[0] for fp in sorted(config.fa_3d_paths) ]  # the order TBSS merge per-subject FA volumes

    if f_name == "DTI_FA":
        X = get_tbss_processed(
            img_path=config.fa_4d_path, 
            mask_path=config.fa_mask_path, 
            N=len(X_subjs), 
            stride=config.downsmple, 
            cache=config.fa_npy_path
        )
    elif f_name == "DTI_MD":
        X = get_tbss_processed(
            img_path=config.md_4d_path, 
            mask_path=config.fa_mask_path, 
            N=len(X_subjs), 
            stride=config.downsmple, 
            cache=config.md_npy_path
        )
    elif f_name in config.sel_feats_by_name.keys():
        sel_feats = config.sel_feats_by_name[f_name]
        X_subjs, X = load_feat_table(config.tbl_paths[f_name], usecols=["BASIC_INFO_ID"]+sel_feats)
    else:
        X_subjs, X = load_feat_table(config.tbl_paths[f_name])
    
    return X_subjs, X


def run_lv1_models(subj_df: pd.DataFrame, config: Config) -> tuple[list[pd.DataFrame], dict]:
    pred_by_feat = []
    summ_by_feat = {}
    src_paths = { **config.tbl_paths, "DTI_FA": config.fa_npy_path, "DTI_MD": config.md_npy_path }

    for f_name in config.feat_types:
        print(f"\nLoading {f_name} data ...")
        X_subjs, X = load_data(f_name, config)
        missing = print_missing(subj_df.index.tolist(), X_subjs)

        ages = subj_df["BASIC_INFO_AGE"].loc[X_subjs].to_numpy(dtype=np.float32)
        sets = subj_df["Set"].loc[X_subjs].to_numpy(dtype=str)
        idx_tr = np.where(sets == "train")[0]
        idx_te = np.where(sets == "test")[0]

        print(f"\nTrain and eval {config.model_lv1} models on {f_name} features ...")
        y_pred, fold_n = train_eval_model(
            X, ages, idx_tr, idx_te, 
            model_type=config.model_lv1, 
            seed=config.seed, 
            seed_inner=config.seed_inner, 
            n_folds=config.n_folds, 
            l1_ratios=config.l1_ratios, 
            alphas=config.alphas, 
            max_iter=config.max_iter, 
            n_jobs=config.n_jobs, 
            verbose=config.verbose, 
            impute_data=False, 
            model_path_template=config.model_paths[f_name], 
            model_perf_path=config.perf_paths[f_name], 
            overwrite=config.overwrite_mdl
        )
        pred_by_feat.append(
            pd.DataFrame({
                "SID": X_subjs,
                f"Fold_{f_name}": fold_n,
                f"Age_{f_name}": y_pred,
            })
        )
        summ_by_feat[f_name] = {
            "source"       : str(src_paths[f_name]),
            "model_paths"  : str(config.model_paths[f_name]),
            "n_subjs"      : X.shape[0],
            "n_features"   : X.shape[1],
            "n_train"      : len(idx_tr),
            "n_test"       : len(idx_te),
            "n_missing"    : len(missing), 
            "missing_subjs": missing,
            "performance"  : pd.read_csv(config.perf_paths[f_name], index_col="Split").to_dict(orient="index")
        }

    return pred_by_feat, summ_by_feat


def merge_preds(subj_df: pd.DataFrame, pred_by_feat: list[pd.DataFrame]) -> pd.DataFrame:
    pred_out = (
        subj_df
        .reset_index()
        .rename(columns={
            subj_df.index.name: "SID",
            "BASIC_INFO_AGE": "Age"
        })
    )
    for df in pred_by_feat:
        pred_out = pd.merge(pred_out, df, on="SID", how="left")

    return pred_out


def add_pymnet_results(pred_out: pd.DataFrame, summ_by_feat: dict, config: Config) -> tuple[pd.DataFrame, dict]:
    f_name = "Pymnet"
    config.feat_types.append(f_name)

    pymnet_df = pd.read_csv(config.pyment_tbl_path, usecols=["subject", "age"])
    pymnet_df.insert(0, "SID", pymnet_df["subject"].map(lambda x: f"sub-{x:04d}").values)
    data_subjs = pymnet_df["SID"].values
    pymnet_df.drop(columns=["subject"], inplace=True)
    pymnet_df.rename(columns={"age": f"Age_{f_name}"}, inplace=True)
    pymnet_df[f"Fold_{f_name}"] = -1  # used publicly available pre-trained model
    pred_out = pd.merge(pred_out, pymnet_df, on="SID", how="left")    
    missing = print_missing(pred_out["SID"].values, data_subjs)

    temp_df = pd.merge(pymnet_df, pred_out.loc[:, ["SID", "Age"]], on="SID", how="left")
    y = temp_df["Age"].values
    y_pred = temp_df[f"Age_{f_name}"].values
    err = y - y_pred
    mae = np.mean(np.abs(err))
    r2 = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)

    summ_by_feat[f_name] = {
        "copied_from"  : str(config.pyment_tbl_path), 
        "n_subjs"      : len(data_subjs), 
        "n_missing"    : len(missing), 
        "missing_subjs": missing,
        "performance"  : {"Split": "All", "MAE": mae, "R2": r2}
    }

    return pred_out, summ_by_feat


def run_lv2_model(pred_out: pd.DataFrame, summ_by_feat: dict, config: Config) -> tuple[pd.DataFrame, dict]:
    lv1_age_cols = [ f"Age_{f_name}" for f_name in config.feat_types ]
    sets = pred_out["Set"].to_numpy(dtype=str)
    idx_tr = np.where(sets == "train")[0]
    idx_te = np.where(sets == "test")[0]

    print(f"\nTrain and eval the final {config.model_lv2} model on the {len(lv1_age_cols)} predicted ages ...")
    y_pred, _ = train_eval_model(
        pred_out[lv1_age_cols].astype(np.float32),  # named, so the fitted model keeps the block names
        pred_out["Age"].to_numpy(dtype=np.float32),
        idx_tr,
        idx_te,
        model_type=config.model_lv2, 
        seed=config.seed, 
        seed_inner=config.seed_inner, 
        n_folds=1, 
        l1_ratios=config.l1_ratios, 
        alphas=config.alphas, 
        max_iter=config.max_iter, 
        n_jobs=config.n_jobs, 
        verbose=config.verbose, 
        impute_data=True, 
        model_path_template=config.lv2_model_path, 
        model_perf_path=config.lv2_perf_path, 
        overwrite=config.overwrite_mdl
    )
    pred_out["Age_Final"] = y_pred

    summ_out = {
        "model_lv1"  : config.model_lv1,
        "model_lv2"  : config.model_lv2,
        "n_folds"    : config.n_folds,
        "seed"       : config.seed,
        "seed_inner" : config.seed_inner,
        "l1_ratios"  : config.l1_ratios,
        "alphas"     : config.alphas,
        "max_iter"   : config.max_iter,
        "n_subjs"    : len(pred_out),
        "n_train"    : len(idx_tr),
        "n_test"     : len(idx_te),
        "predictions": str(config.pred_out_path),
        "feat_types" : summ_by_feat,
        "final"      : {
            "model_paths": str(config.lv2_model_path),
            "features"   : lv1_age_cols,
            "performance": pd.read_csv(config.lv2_perf_path, index_col="Split").to_dict(orient="index")
        }
    }

    return pred_out, summ_out


def main(config: Config):
    subj_df = pd.read_csv(config.df_preproc_path, index_col="BASIC_INFO_ID", usecols=["BASIC_INFO_ID", "BASIC_INFO_AGE", "Set"])
    n_subjs = len(subj_df)
    print(f"\nNumber of participants: {n_subjs}")

    pred_by_feat, summ_by_feat = run_lv1_models(subj_df, config)
    pred_out = merge_preds(subj_df, pred_by_feat)
    pred_out, summ_by_feat = add_pymnet_results(pred_out, summ_by_feat, config)    
    pred_out, summ_out = run_lv2_model(pred_out, summ_by_feat, config)

    config.pred_out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_csv(config.pred_out_path, index=False)
    print(f"\nSaved: {config.pred_out_path}\n")

    config.summ_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.summ_out_path, "w") as f:
        json.dump(to_json_compatible(summ_out), f, allow_nan=False)
    print(f"Saved: {config.summ_out_path}\n")


if __name__ == "__main__":
    config = Config(parse_args())
    log_path = Path(config.log_out_path)

    with tee_output(log_path):
        print(f"\nStart at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main(config)
        print(f"\nFinish at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nDone! logs is saved to: {log_path}")
