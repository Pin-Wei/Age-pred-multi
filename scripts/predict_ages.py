#!/usr/bin/env python3


import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import (
    MODEL_TYPES, L1_RATIOS, ALPHAS, 
    print_missing, get_tbss_processed, load_feat_table, train_eval_model
) 
from utils import to_json_compatible, tee_output


SID, SET, AGE, COG = "SID", "Set", "Age", "Cog"  # column names
TARGETS = [[AGE], [AGE, COG]]


class Config:
    def __init__(self, args: argparse.Namespace | None = None):
        args = parse_args([]) if args is None else args
        self.setup_model_params(args)
        self.setup_vars_and_paths(args)
        self.only_run_lv1_models = args.only_run_lv1_models

    def setup_model_params(self, args):
        self.targets = TARGETS[args.targets]
        self.model_lv1 = args.model_lv1
        self.model_lv2 = args.model_lv2
        self.l1_ratios = list(args.l1_ratios)
        self.alphas = list(args.alphas)
        self.n_folds = args.n_folds
        self.seed = args.seed
        self.seed_inner = args.seed_inner
        self.max_iter = args.max_iter
        self.xgb_params = {
            # "max_depth"       : 2,
            # "learning_rate"   : 0.06,
            # "n_estimators"    : 500,
            # "min_child_weight": 8,
            # "subsample"       : 0.7,
            # "colsample_bytree": 0.5, 
            "max_bin"         : 64
        }
        self.opt_trials = args.opt_trials
        self.n_jobs = args.n_jobs
        self.xgb_device = args.xgb_device
        self.verbose = args.verbose
        self.overwrite_mdl = args.overwrite_mdl

    def setup_vars_and_paths(self, args):
        self.proj_root = Path(__file__).resolve().parents[1]
        self.tbl_dir = self.proj_root / "data" / "tabular"
        self.cog_score_path = self.tbl_dir / "cog_scores.json"
        self.pyment_tbl_path = self.proj_root / "data" / "pyment" / "predictions" / "predictions.csv"
        
        lv1_key = f"{len(self.targets)}y_{self.model_lv1}_cv{self.n_folds}_{self.seed}"
        if args.add_new_mdls:
            while (self.proj_root / "models" / lv1_key).is_dir():
                lv1_key += "+"
        self.lv1_mdl_dir = self.proj_root / "models" / lv1_key
        self.lv1_res_dir = self.proj_root / "results" / lv1_key
        
        lv2_key = f"{self.model_lv2}_{self.seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.lv2_mdl_dir = self.lv1_mdl_dir / lv2_key
        self.lv2_res_dir = self.lv1_res_dir / lv2_key
        self.pred_out_path = self.lv2_res_dir / "predictions.csv"
        self.summ_out_path = self.lv2_res_dir / "summary.json"
        self.log_out_path  = self.lv2_res_dir / "logs.txt"

        preproc_f_names = self._for_preproc()
        tbss_key = self._for_tbss(args)
        self._for_models(args, preproc_f_names, tbss_key)

    def _for_preproc(self):
        self.df_preproc_path = self.tbl_dir / "df_preproc.csv"
        preproc_feats_path = self.proj_root / "data" / "meta" / "preproc_features.txt"

        feats = preproc_feats_path.read_text().splitlines()
        self.sel_feats_by_name = {
            "MRI_ROI": [ f for f in feats if f.startswith("STRUCTURE") and not f.endswith("FA") ]
        }

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
        # for approach, domain in product(["MRI", "EEG", "BEH"], ["LANGUAGE", "MEMORY", "MOTOR"]):
        #     f_name = f"{approach}_{domain}"
        #     sel_feats = [ f for f in feats if f.startswith(domain) and approach in f ]
        #     if sel_feats:
        #         self.sel_feats_by_name.update({f_name: sel_feats})

        return list(self.sel_feats_by_name.keys())

    def _for_tbss(self, args):
        self.downsmple = args.downsample
        skeleton = int(args.skeleton)

        tbss_dir          = self.proj_root / "data" / "dti" / "tbss"
        self.fa_3d_dir    = tbss_dir / "origdata"
        self.fa_4d_path   = tbss_dir / "stats" / f"all_FA{['', '_skeletonised'][skeleton]}.nii.gz"
        self.md_4d_path   = tbss_dir / "stats" / f"all_MD{['', '_skeletonised'][skeleton]}.nii.gz"
        self.fa_mask_path = tbss_dir / "stats" / f"mean_FA{['', '_skeleton'][skeleton]}_mask.nii.gz"
        
        key = f"{['', '-sk'][skeleton]}_ds{self.downsmple}"
        self.fa_npy_path = self.proj_root / "data" / "dti" / f"tbss_FA{key}_flat.npy"
        self.md_npy_path = self.proj_root / "data" / "dti" / f"tbss_MD{key}_flat.npy"
        
        return key

    def _for_models(self, args, preproc_f_names, tbss_key):
        feat_name_2_affix = {
            "DTI_FA"    : f"dti_fa{tbss_key}", 
            "DTI_MD"    : f"dti_md{tbss_key}", 
            # "DTI_ROI"   : "dti_roi-means", 
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
        if args.feat_types:  # the paths below stay complete; only the first level's loop shrinks
            unknown = [ f for f in args.feat_types if f not in feat_name_2_affix ]
            assert not unknown, f"\nUnknown --feat_types {unknown}; available: {self.feat_types}\n"
            self.feat_types = [ f for f in self.feat_types if f in set(args.feat_types) ]

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

            self.model_paths[f_name] = self.lv1_mdl_dir / f"{affix}_{{}}.joblib"
            self.perf_paths[f_name]  = self.lv1_res_dir / f"{affix}.csv"

        self.lv2_model_path = self.lv2_mdl_dir / f"pipeline_{{}}.joblib"
        self.lv2_perf_path  = self.lv2_res_dir / f"performance.csv"
        self.lv2_calib_path = self.lv2_res_dir / f"age-correction_params.csv"


def parse_args(
    argv: list[str] = None, 
    parser: argparse.ArgumentParser = None, 
    defaults: dict = None
) -> argparse.Namespace:
    '''
    Command-line overrides for the Config values worth varying between runs.

    A downstream script sharing this Config passes its own `parser`, 
    already holding its extra options, 
    so that these options are added to it rather than to a fresh one. 
    
    It may also pass `defaults` to replace the ones set below, 
    which `argv` still overrides.
    '''
    parser = parser or argparse.ArgumentParser(
        description="Train and evaluate the two-level age-prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--only_run_lv1_models", action="store_true", 
                        help="Only run the first-order models; " + 
                             "may be useful when you want to run the script in parallel on a different server.")

    grp_model = parser.add_argument_group("model")
    grp_model.add_argument("--targets", type=int, choices=range(len(TARGETS)), default=1, 
                           help="The target(s) for models to predict. " + 
                                ", ".join([ f'{i}: {x}' for i, x in enumerate(TARGETS) ]))
    grp_model.add_argument("--model_lv1", choices=MODEL_TYPES, default=MODEL_TYPES[0],
                           help="Type of regression algorithm to use for first-level models")
    grp_model.add_argument("--model_lv2", choices=MODEL_TYPES, default=MODEL_TYPES[2],
                           help="Type of regression algorithm to use for the second-level model")
    grp_model.add_argument("--l1_ratios", nargs="+", type=float, default=L1_RATIOS, metavar="R",
                           help="L1 penalty mixing parameter grid for ElasticNet models")
    grp_model.add_argument("--alphas", nargs="+", type=float, default=ALPHAS, metavar="A",
                           help="Regularization strength grid for linear models")
    grp_model.add_argument("--n_folds", type=int, default=5,
                           help="Number of outer CV folds for the first-level models")
    grp_model.add_argument("--max_iter", type=int, default=10000, 
                           help="Maximum number of iterations for the linear solvers")
    grp_model.add_argument("--opt_trials", type=int, default=100, 
                           help="Number of Optuna trials spent on searching the best XGBoost hyperparameters")
    grp_model.add_argument("--seed", type=int, default=42,
                           help="Random seed for the outer cross-validation splits")
    grp_model.add_argument("--seed_inner", type=int, default=0, 
                           help="Random seed for hyperparameter tuning")

    grp_data = parser.add_argument_group("data")
    grp_data.add_argument("--feat_types", nargs="+", default=None, metavar="FEAT",
                          help="Run first-level models only for specified feature types")
    grp_data.add_argument("--downsample", type=int, default=2,
                          help="Voxel stride for the TBSS volumes")
    grp_data.add_argument("--skeleton", action="store_true",
                          help="Use the skeletonised TBSS volumes instead of the full ones")

    grp_run = parser.add_argument_group("runtime")
    grp_run.add_argument("--n_jobs", type=int, default=12, 
                         help="Number of CPU cores used for parallel execution; -1 uses all available")
    grp_run.add_argument("--xgb_device", choices=["cpu", "cuda"], default="cuda", 
                         help="Device on which the XGBoost models are trained; a fit that runs out of GPU memory is repeated on the CPU")
    grp_run.add_argument("--verbose", type=int, default=1, 
                         help="Verbosity level of execution logs")
    grp_run.add_argument("--overwrite_mdl", action="store_true",
                         help="Refit and overwrite the models already saved on disk")
    grp_run.add_argument("--add_new_mdls", action="store_true", 
                         help="Create a new folder and retrain the models without overwriting the existing ones.")

    parser.set_defaults(**(defaults or {}))  # after the options exist, so that --help shows the replacements

    return parser.parse_args(argv)


def load_targets(config: Config) -> pd.DataFrame:
    subj_df = pd.read_csv(
        config.df_preproc_path, index_col="BASIC_INFO_ID", 
        usecols=["BASIC_INFO_ID", "BASIC_INFO_AGE", SET]
    )
    subj_df.index.name = SID
    subj_df = subj_df.rename(columns={"BASIC_INFO_AGE": AGE})

    if COG in config.targets:
        assert config.cog_score_path.is_file(), f"\n'{config.cog_score_path.name}' not exists; run make_df_scores.py first\n"
        cog_data = json.loads(config.cog_score_path.read_text())
        cog_scores = pd.Series(cog_data["scores"])

        print_missing(subj_df.index.to_list(), cog_scores.index.to_list())
        subj_df[COG] = cog_scores

    return subj_df.loc[:, [SET] + config.targets]


def load_data(f_name: str, config: Config) -> tuple[list[str], np.ndarray | pd.DataFrame]:
    if f_name == "DTI_FA":
        return get_tbss_processed(
            img_path=config.fa_4d_path, 
            subj_dir=config.fa_3d_dir, 
            stride=config.downsmple, 
            mask_path=config.fa_mask_path, 
            cache=config.fa_npy_path
        )
    elif f_name == "DTI_MD":
        return get_tbss_processed(
            img_path=config.md_4d_path, 
            subj_dir=config.fa_3d_dir, 
            stride=config.downsmple, 
            mask_path=config.fa_mask_path, 
            cache=config.md_npy_path
        )
    elif f_name in config.sel_feats_by_name.keys():
        sel_feats = config.sel_feats_by_name[f_name]
        return load_feat_table(config.tbl_paths[f_name], usecols=["BASIC_INFO_ID"] + sel_feats)
    else:
        return load_feat_table(config.tbl_paths[f_name])


def run_lv1_models(subj_df: pd.DataFrame, config: Config) -> tuple[list[pd.DataFrame], dict]:
    pred_by_feat = []
    summ_by_feat = {}
    src_paths = { **config.tbl_paths, "DTI_FA": config.fa_npy_path, "DTI_MD": config.md_npy_path }

    for f_name in config.feat_types:
        print(f"\nLoading {f_name} data ...")
        subj_list, X = load_data(f_name, config)
        missing = print_missing(subj_df.index.to_list(), subj_list)

        y = subj_df.loc[subj_list, config.targets].astype(np.float32)
        sets = subj_df.loc[subj_list, SET].to_numpy(dtype=str)
        idx_tr = np.where(sets == "train")[0]
        idx_te = np.where(sets == "test")[0]

        print(f"\nTrain and eval {config.model_lv1} models on {f_name} features ...")
        y_pred, _, fold_n = train_eval_model(  # no age-bias correction on this level
            X, y, idx_tr, idx_te, 
            model_type=config.model_lv1, 
            seed=config.seed, 
            seed_inner=config.seed_inner, 
            n_folds=config.n_folds, 
            l1_ratios=config.l1_ratios, 
            alphas=config.alphas, 
            max_iter=config.max_iter, 
            xgb_params=config.xgb_params, 
            opt_trials=config.opt_trials, 
            n_jobs=config.n_jobs, 
            xgb_device=config.xgb_device, 
            verbose=config.verbose, 
            impute_data=False, 
            model_path_template=config.model_paths[f_name], 
            model_perf_path=config.perf_paths[f_name], 
            overwrite=config.overwrite_mdl
        )
        pred_by_feat.append(
            pd.concat([
                pd.DataFrame({SID: subj_list, f"Fold_{f_name}": fold_n}), 
                y_pred.add_suffix(f"_{f_name}")
            ], axis=1)
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
    pred_out = subj_df.reset_index()

    for df in pred_by_feat:
        pred_out = pd.merge(pred_out, df, on=SID, how="left")

    return pred_out


def add_pyment_results(pred_out: pd.DataFrame, summ_by_feat: dict, config: Config) -> tuple[pd.DataFrame, dict]:
    f_name = "Pyment"
    config.feat_types.append(f_name)

    pyment_df = pd.read_csv(config.pyment_tbl_path, usecols=["subject", "age"])
    pyment_df.insert(0, SID, pyment_df["subject"].map(lambda x: f"sub-{x:04d}").values)
    data_subjs = pyment_df[SID].values
    pyment_df.drop(columns=["subject"], inplace=True)
    pyment_df.rename(columns={"age": f"{AGE}_{f_name}"}, inplace=True)
    pyment_df[f"Fold_{f_name}"] = -1  # used publicly available pre-trained model
    pred_out = pd.merge(pred_out, pyment_df, on=SID, how="left")    
    missing = print_missing(pred_out[SID].values, data_subjs)

    temp_df = pd.merge(pyment_df, pred_out.loc[:, [SID, AGE]], on=SID, how="left")
    y = temp_df[AGE].values
    y_pred = temp_df[f"{AGE}_{f_name}"].values
    err = y - y_pred
    mae = np.mean(np.abs(err))
    r2 = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)

    summ_by_feat[f_name] = {
        "copied_from"  : str(config.pyment_tbl_path), 
        "n_subjs"      : len(data_subjs), 
        "n_missing"    : len(missing), 
        "missing_subjs": missing,
        "performance"  : {"Split": "All", f"MAE_{AGE}": mae, f"R2_{AGE}": r2}
    }

    return pred_out, summ_by_feat


def run_lv2_model(pred_out: pd.DataFrame, summ_by_feat: dict, config: Config) -> tuple[pd.DataFrame, dict]:
    lv1_pred_cols = [ 
        f"{target}_{f_name}" 
        for f_name in config.feat_types for target in config.targets 
        if f"{target}_{f_name}" in pred_out.columns
    ]
    sets = pred_out[SET].to_numpy(dtype=str)
    idx_tr = np.where(sets == "train")[0]
    idx_te = np.where(sets == "test")[0]

    print(f"\nTrain and eval the final {config.model_lv2} model on the {len(lv1_pred_cols)} first-level prediction(s) ...")
    y_pred, y_pred_ac, _ = train_eval_model(
        X=pred_out.loc[:, lv1_pred_cols].astype(np.float32), 
        y=pred_out.loc[:, config.targets].astype(np.float32),
        idx_tr=idx_tr,
        idx_te=idx_te,
        model_type=config.model_lv2,
        seed=config.seed,
        seed_inner=config.seed_inner,
        n_folds=config.n_folds, 
        l1_ratios=config.l1_ratios,
        alphas=config.alphas,
        max_iter=config.max_iter, 
        xgb_params=config.xgb_params, 
        opt_trials=config.opt_trials, 
        n_jobs=config.n_jobs, 
        xgb_device=config.xgb_device, 
        verbose=config.verbose,
        impute_data=True,
        apply_correction=True,
        model_path_template=config.lv2_model_path,
        model_perf_path=config.lv2_perf_path,
        calib_param_path=config.lv2_calib_path, 
        overwrite=config.overwrite_mdl
    )
    pred_out = pd.concat([
        pred_out, 
        y_pred.add_suffix("_Final"), 
        y_pred_ac.add_suffix("_Final").add_prefix("C-")
    ], axis=1)

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
            "features"   : lv1_pred_cols,
            "performance": pd.read_csv(config.lv2_perf_path, index_col="Split").to_dict(orient="index")
        }
    }

    return pred_out, summ_out


def main(config: Config):
    subj_df = load_targets(config)
    print(f"\nNumber of participants: {len(subj_df)}")

    pred_by_feat, summ_by_feat = run_lv1_models(subj_df, config)
    if config.only_run_lv1_models:
        exit()

    pred_out = merge_preds(subj_df, pred_by_feat)
    pred_out, summ_by_feat = add_pyment_results(pred_out, summ_by_feat, config)    
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

    print(f"\nDone! logs is saved to: {log_path}\n")
