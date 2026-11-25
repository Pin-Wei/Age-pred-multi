#!/usr/bin/env python3


import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import get_tbss_processed, load_feat_table, print_missing
from training import (
    MODEL_TYPES, L1_RATIOS, ALPHAS, ALPHAS_LOWER, DEFAULT_PLS_MAX_K,
    calc_metrics, train_eval_model,
)
from utils import custom_print, tee_output, to_json_compatible


SID, SET, AGE, COG, MIX = "SID", "Set", "Age", "Cog", "Mix"  # column names
TARGETS = [[AGE], [AGE, COG], [MIX]]
TARGET_KEYS = ["1y", "2y", "mix"]  # prefix of the first-level folder, one per entry of TARGETS
LV1_OUTPUTS = ["targ-preds", "cross-decomp"]  # output type of the first-level models; target(s) predicted values or the components derived from cross-decomposition
DECOMP_MDL = "pls"  # the model (algorithm) used for cross-decomposition

TBSS_MODALITIES = {"DTI_FA", "DTI_MD"}
COGNITIVE_DOMAINS = ("LANGUAGE", "MEMORY", "MOTOR")
FUNCTIONAL_APPROACHES = ("MRI", "EEG")


class Config:
    '''
    Experiment configuration and path resolver shared with downstream scripts.
    Runtime outputs (such as resolved level-1 column names) are passed explicitly
    between stages rather than mutated on this object.
    '''
    def __init__(self, args: argparse.Namespace | None = None):
        args = parse_args([]) if args is None else args
        self.setup_model_params(args)
        self.setup_vars_and_paths(args)
        self.only_run_lv1_models = args.only_run_lv1_models

    def setup_model_params(self, args: argparse.Namespace) -> None:
        self.targets = TARGETS[args.targets]
        self.mix_in_years = args.mix_in_years and (MIX in self.targets)
        self.target_key = TARGET_KEYS[args.targets] + ("a" if self.mix_in_years else "")  # "mixa": age scale

        self.lv1_output = args.lv1_output
        self.decomp = self.lv1_output == "cross-decomp"
        if self.decomp and (args.model_lv1 not in {DECOMP_MDL, MODEL_TYPES[0]}):
            custom_print(
                f"\n--model_lv1 {args.model_lv1} is ignored: "
                f"the '{self.lv1_output}' route decomposes features with {DECOMP_MDL.upper()}.\n",
                level="WARNING",
            )

        self.model_lv1 = DECOMP_MDL if self.decomp else args.model_lv1
        self.pls_components = (
            (DEFAULT_PLS_MAX_K if self.decomp else "cv")
            if args.pls_components is None
            else args.pls_components
        )
        self.pls_max_components = args.pls_max_components
        self.pls_n_perm = args.pls_n_perm

        self.model_lv2 = args.model_lv2
        self.l1_ratios = list(args.l1_ratios)

        # Use a lower-scale alpha grid by default for z-scored Mix unless --alphas was explicitly given
        if args.alphas is None:
            if (MIX in self.targets) and not self.mix_in_years:
                self.alphas = list(ALPHAS_LOWER)
            else:
                self.alphas = list(ALPHAS)
        else:
            self.alphas = list(args.alphas)

        self.n_folds = args.n_folds
        self.seed = args.seed
        self.seed_inner = args.seed_inner
        self.max_iter = args.max_iter
        self.xgb_params = {
            # "max_depth": ,
            # "learning_rate": ,
            # "n_estimators": ,
            # "min_child_weight": ,
            # "subsample": ,
            # "colsample_bytree": ,
            "max_bin": 64,
        }
        self.opt_trials = args.opt_trials
        self.n_jobs = args.n_jobs
        self.xgb_device = args.xgb_device
        self.verbose = args.verbose
        self.overwrite_mdl = args.overwrite_mdl

    def setup_vars_and_paths(self, args: argparse.Namespace) -> None:
        self.proj_root       = Path(__file__).resolve().parents[1]
        self.tbl_dir         = self.proj_root / "data" / "tabular"
        self.df_preproc_path = self.tbl_dir / "df_preproc.csv"
        self.cog_score_path  = self.tbl_dir / "cog_scores.json"
        self.pyment_tbl_path = self.proj_root / "data" / "pyment" / "predictions" / "predictions.csv"

        if self.decomp:
            comp_tag = (
                f"comp{self.pls_components}"
                if isinstance(self.pls_components, int)
                else f"comp-{self.pls_components}"
            )
            mdl_tag = f"{self.model_lv1}-{comp_tag}"  # holds no "_": downstream scripts split on "_"
        elif self.model_lv1 == "pls" and self.pls_components != "cv":
            k_tag = f"k{self.pls_components}" if isinstance(self.pls_components, int) else str(self.pls_components)
            mdl_tag = f"{self.model_lv1}-{k_tag}"
        else:
            mdl_tag = self.model_lv1

        lv1_key = f"{self.target_key}_{mdl_tag}_cv{self.n_folds}_{self.seed}"
        if args.add_new_mdls:
            while (self.proj_root / "models" / lv1_key).is_dir() or (self.proj_root / "results" / lv1_key).is_dir():
                lv1_key += "+"

        self.lv1_mdl_dir = self.proj_root / "models" / lv1_key
        self.lv1_res_dir = self.proj_root / "results" / lv1_key

        lv2_key = f"{self.model_lv2}_{self.seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.lv2_mdl_dir   = self.lv1_mdl_dir / lv2_key
        self.lv2_res_dir   = self.lv1_res_dir / lv2_key
        self.pls_feat_path = self.lv2_res_dir / "pls_feats.csv"
        self.pred_out_path = self.lv2_res_dir / "predictions.csv"
        self.summ_out_path = self.lv2_res_dir / "summary.json"
        self.log_out_path  = self.lv2_res_dir / "logs.txt"

        self.lv2_model_path = self.lv2_mdl_dir / "pipeline_{}.joblib"
        self.lv2_perf_path  = self.lv2_res_dir / "performance.csv"
        self.lv2_calib_path = self.lv2_res_dir / "age-correction_params.csv"

        # TBSS paths
        self.downsample = args.downsample
        skel_suffix = "_skeletonised" if args.skeleton else ""
        mask_suffix = "_skeleton" if args.skeleton else ""
        tbss_key = f"{'-sk' if args.skeleton else ''}_ds{self.downsample}"

        tbss_dir          = self.proj_root / "data" / "dti" / "tbss"
        self.fa_3d_dir    = tbss_dir / "origdata"
        self.fa_4d_path   = tbss_dir / "stats" / f"all_FA{skel_suffix}.nii.gz"
        self.md_4d_path   = tbss_dir / "stats" / f"all_MD{skel_suffix}.nii.gz"
        self.fa_mask_path = tbss_dir / "stats" / f"mean_FA{mask_suffix}_mask.nii.gz"
        self.fa_npy_path  = self.proj_root / "data" / "dti" / f"tbss_FA{tbss_key}_flat.npy"
        self.md_npy_path  = self.proj_root / "data" / "dti" / f"tbss_MD{tbss_key}_flat.npy"

        # Preprocessed feature groups & per-modality file mappings
        self.sel_feats_by_name = self._parse_preproc_groups(
            self.proj_root / "data" / "meta" / "preproc_features.txt"
        )
        affix_by_feat = {
            "DTI_FA"    : f"dti_fa{tbss_key}",
            "DTI_MD"    : f"dti_md{tbss_key}",
            # "DTI_ROI"   : "dti_roi-means",
            "rs-EEG_PSD": "rs-eeg_psd",
            "rs-MRI_FC" : "rs-fmri_fc",
            "MRI_ROI"   : "t1w_roi-means",
            **{ name: name.lower() for name in self.sel_feats_by_name if name != "MRI_ROI" },
        }

        self.feat_types = list(affix_by_feat.keys())
        if args.feat_types:
            unknown = [ f for f in args.feat_types if f not in affix_by_feat ]
            if unknown:
                raise ValueError(f"Unknown --feat_types {unknown}; available: {self.feat_types}")
            requested = set(args.feat_types)
            self.feat_types = [ f for f in self.feat_types if f in requested ]

        self.tbl_paths: dict[str, Path | None] = {}
        self.model_paths: dict[str, Path] = {}
        self.perf_paths: dict[str, Path] = {}

        for f_name, affix in affix_by_feat.items():
            if f_name in TBSS_MODALITIES:
                self.tbl_paths[f_name] = None
            elif f_name in self.sel_feats_by_name:
                self.tbl_paths[f_name] = self.df_preproc_path
            else:
                self.tbl_paths[f_name] = self.tbl_dir / f"df_{affix}.csv"

            self.model_paths[f_name] = self.lv1_mdl_dir / f"{affix}_{{}}.joblib"
            self.perf_paths[f_name]  = self.lv1_res_dir / f"{affix}.csv"

    @staticmethod
    def _parse_preproc_groups(feat_file: Path) -> dict[str, list[str]]:
        feats = feat_file.read_text().splitlines()
        cog_feats = [ f for f in feats if f.startswith(COGNITIVE_DOMAINS) ]

        return {
            "MRI_ROI"      : [ f for f in feats if f.startswith("STRUCTURE") and not f.endswith("FA") ],
            "FUN_COGNITIVE": [ f for f in cog_feats if any(app in f for app in FUNCTIONAL_APPROACHES) ],
            "BEH_COGNITIVE": [ f for f in cog_feats if "BEH" in f ],
        }

    def common_train_kwargs(self, model_type: str) -> dict:
        '''
        Shared keyword arguments passed to `train_eval_model` for both levels.
        '''
        return {
            "model_type"        : model_type,
            "seed"              : self.seed,
            "seed_inner"        : self.seed_inner,
            "n_folds"           : self.n_folds,
            "l1_ratios"         : self.l1_ratios,
            "alphas"            : self.alphas,
            "max_iter"          : self.max_iter,
            "xgb_params"        : self.xgb_params,
            "opt_trials"        : self.opt_trials,
            "pls_components"    : self.pls_components if model_type == "pls" else None,
            "pls_max_components": self.pls_max_components,
            "pls_n_perm"        : self.pls_n_perm,
            "n_jobs"            : self.n_jobs,
            "xgb_device"        : self.xgb_device,
            "verbose"           : self.verbose,
            "overwrite"         : self.overwrite_mdl,
        }

    def summary_metadata(self) -> dict:
        '''
        Top-level run metadata recorded in `summary.json`.
        '''
        return {
            "targets"           : self.targets,
            "mix_in_years"      : self.mix_in_years,
            "lv1_output"        : self.lv1_output,
            "pls_components"    : self.pls_components if self.decomp else None,
            "pls_max_components": self.pls_max_components if self.decomp else None,
            "pls_n_perm"        : self.pls_n_perm if (self.decomp and self.pls_components == "auto") else None,
            "model_lv1"         : self.model_lv1,
            "model_lv2"         : self.model_lv2,
            "n_folds"           : self.n_folds,
            "seed"              : self.seed,
            "seed_inner"        : self.seed_inner,
            "l1_ratios"         : self.l1_ratios,
            "alphas"            : self.alphas,
            "max_iter"          : self.max_iter,
        }


def parse_args(
    argv: list[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
    defaults: dict | None = None,
) -> argparse.Namespace:
    '''
    Command-line overrides for the Config values worth varying between runs.
    '''
    def _parse_pls_components(val: str) -> int | str:
        low = val.strip().lower()
        if low in {"auto", "cv"}:
            return low

        try:
            k = int(val)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--pls_components must be a positive integer, 'auto', or 'cv'; got {val!r}."
            ) from exc

        if k < 1:
            raise argparse.ArgumentTypeError(f"--pls_components must be >= 1; got {k}.")

        return k

    parser = parser or argparse.ArgumentParser(
        description="Train and evaluate the two-level age-prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--only_run_lv1_models", action="store_true", 
                        help="Only run the first-order models; useful when running in parallel on a different server.")

    grp_model = parser.add_argument_group("model")
    grp_model.add_argument("--targets", type=int, choices=range(len(TARGETS)), default=1, 
                           help="Target(s) for models to predict. " + ", ".join(f"{i}: {x}" for i, x in enumerate(TARGETS)))
    grp_model.add_argument("--mix_in_years", action="store_true", 
                           help=("Express the composite target in years rather than in units of its own std "
                                 f"(applies to --targets {TARGETS.index([MIX])} only)."))
    grp_model.add_argument("--lv1_output", choices=LV1_OUTPUTS, default=LV1_OUTPUTS[0], 
                           help=("What first-level models pass to level 2: 'targ-preds' (predicted targets) or "
                                 f"'cross-decomp' ({DECOMP_MDL.upper()} latent components)."))
    grp_model.add_argument("--pls_components", type=_parse_pls_components, default=None, metavar="K|auto|cv", 
                           help=("PLS component selection: an integer K, 'auto' (OOF parallel analysis), or 'cv' (inner CV). "
                                 f"None defaults to {DEFAULT_PLS_MAX_K} under 'cross-decomp' and 'cv' under 'targ-preds'."))
    grp_model.add_argument("--pls_max_components", type=int, default=DEFAULT_PLS_MAX_K, metavar="M",
                           help="Upper bound on PLS components when --pls_components is 'auto' or 'cv'.")
    grp_model.add_argument("--pls_n_perm", type=int, default=100, metavar="B",
                           help="Number of Y-permutations used by parallel analysis (--pls_components auto).")
    grp_model.add_argument("--model_lv1", choices=MODEL_TYPES, default=MODEL_TYPES[0],
                           help="Regression algorithm for first-level models; ignored under '--lv1_output cross-decomp'.")
    grp_model.add_argument("--model_lv2", choices=MODEL_TYPES, default=MODEL_TYPES[2],
                           help="Regression algorithm for the second-level model.")
    grp_model.add_argument("--l1_ratios", nargs="+", type=float, default=L1_RATIOS, metavar="R",
                           help="L1 penalty mixing parameter grid for ElasticNet models.")
    grp_model.add_argument("--alphas", nargs="+", type=float, default=None, metavar="A",
                           help="Regularization strength grid for linear models; None picks ALPHAS (or ALPHAS_LOWER for z-scored Mix).")
    grp_model.add_argument("--n_folds", type=int, default=5,
                           help="Number of outer CV folds.")
    grp_model.add_argument("--max_iter", type=int, default=10000,
                           help="Maximum number of iterations for the linear solvers.")
    grp_model.add_argument("--opt_trials", type=int, default=100,
                           help="Number of Optuna trials spent searching XGBoost hyperparameters.")
    grp_model.add_argument("--seed", type=int, default=42,
                           help="Random seed for the outer cross-validation splits.")
    grp_model.add_argument("--seed_inner", type=int, default=0,
                           help="Random seed for hyperparameter tuning / parallel analysis.")

    grp_data = parser.add_argument_group("data")
    grp_data.add_argument("--feat_types", nargs="+", default=None, metavar="FEAT",
                          help="Run first-level models only for specified feature types.")
    grp_data.add_argument("--downsample", type=int, default=2,
                          help="Voxel stride for the TBSS volumes.")
    grp_data.add_argument("--skeleton", action="store_true",
                          help="Use the skeletonised TBSS volumes instead of the full ones.")

    grp_run = parser.add_argument_group("runtime")
    grp_run.add_argument("--n_jobs", type=int, default=12,
                         help="Number of CPU cores used for parallel execution; -1 uses all available.")
    grp_run.add_argument("--xgb_device", choices=["cpu", "cuda"], default="cuda",
                         help="Device on which XGBoost models are trained; an OOM fit is repeated on CPU.")
    grp_run.add_argument("--verbose", type=int, default=1,
                         help="Verbosity level of execution logs.")
    grp_run.add_argument("--overwrite_mdl", action="store_true",
                         help="Refit and overwrite models already saved on disk.")
    grp_run.add_argument("--add_new_mdls", action="store_true",
                         help="Create a new folder and retrain models without overwriting existing ones.")

    parser.set_defaults(**(defaults or {}))

    return parser.parse_args(argv)


def split_indices(sets: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sets, dtype=str)

    return np.flatnonzero(arr == "train"), np.flatnonzero(arr == "test")


def make_mix(subj_df: pd.DataFrame, in_years: bool = False) -> pd.Series:
    '''
    Combine Age and Cog into one equal-weight composite target standardized on training rows.
    Cog enters with the sign of its training correlation with Age so the two do not cancel out.
    '''
    tr_idx, _ = split_indices(subj_df[SET])
    age_tr = subj_df[AGE].iloc[tr_idx]
    cog_tr = subj_df[COG].iloc[tr_idx]

    age_mean, age_std = float(age_tr.mean()), float(age_tr.std())
    cog_mean, cog_std = float(cog_tr.mean()), float(cog_tr.std())
    sign = -1.0 if float(age_tr.corr(cog_tr)) < 0 else 1.0

    z_age = (subj_df[AGE] - age_mean) / age_std
    z_cog = (subj_df[COG] - cog_mean) / cog_std
    mix = (z_age + sign * z_cog) / 2.0
    formula = f"[ z({AGE}) {'+' if sign > 0 else '-'} z({COG}) ] / 2"

    if in_years:
        mix = age_mean + age_std * mix
        formula = f"{age_mean:.3f} + {age_std:.3f} * {formula}"

    custom_print(f"\n'{MIX}' = {formula}, standardized on the {len(tr_idx)} training participants")

    return mix


def restore_from_mix(pred_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Map every composite-target prediction back to Age and Cog via least-squares fits
    estimated on the training participants, so downstream scripts find Age_* and Cog_*.
    '''
    is_train = pred_df[SET].to_numpy(dtype=str) == "train"
    mix_cols = [ c for c in pred_df.columns if c.startswith((f"{MIX}_", f"C-{MIX}_")) ]
    restored = {}

    for target in (AGE, COG):
        rows = is_train & pred_df[[MIX, target]].notna().all(axis=1).to_numpy()
        slope, const = np.polyfit(pred_df.loc[rows, MIX], pred_df.loc[rows, target], 1)
        for col in mix_cols:
            name = col.replace(MIX, target, 1)  # replace only the leading target prefix
            if name not in pred_df.columns:
                restored[name] = const + slope * pred_df[col]
        custom_print(
            f"{target} = {const:.3f} + {slope:.3f} * {MIX}, fitted on {int(rows.sum())} training participants"
        )

    return pd.concat([pred_df, pd.DataFrame(restored, index=pred_df.index)], axis=1)


def load_targets(config: Config) -> pd.DataFrame:
    subj_df = pd.read_csv(
        config.df_preproc_path,
        index_col="BASIC_INFO_ID",
        usecols=["BASIC_INFO_ID", "BASIC_INFO_AGE", SET],
    )
    subj_df.index.name = SID
    subj_df = subj_df.rename(columns={"BASIC_INFO_AGE": AGE})

    need_cog = (COG in config.targets) or (MIX in config.targets)
    if need_cog:
        if not config.cog_score_path.is_file():
            raise FileNotFoundError(f"'{config.cog_score_path.name}' does not exist; run make_df_scores.py first.")
        
        cog_scores = pd.Series(json.loads(config.cog_score_path.read_text())["scores"])
        print_missing(subj_df.index.to_list(), cog_scores.index.to_list())
        subj_df[COG] = cog_scores

    req_cols = [AGE, COG] if need_cog else [AGE]
    n_before = len(subj_df)
    subj_df = subj_df.dropna(subset=req_cols)
    n_dropped = n_before - len(subj_df)
    if n_dropped > 0:
        custom_print(
            f"\nDropped {n_dropped} participant(s) with missing target value(s) in {req_cols}.",
            level="WARNING",
        )

    if MIX in config.targets:
        subj_df[MIX] = make_mix(subj_df, config.mix_in_years)
        return subj_df.loc[:, [SET, AGE, COG, MIX]]

    return subj_df.loc[:, [SET] + config.targets]


def load_preproc_cache(config: Config) -> tuple[np.ndarray, pd.DataFrame] | None:
    '''
    Read `df_preproc` once for all requested preprocessed feature groups
    (MRI_ROI, FUN_COGNITIVE, BEH_COGNITIVE) to avoid redundant disk I/O.
    '''
    active_groups = [ f for f in config.feat_types if f in config.sel_feats_by_name ]
    if not active_groups:
        return None

    needed_cols = list(dict.fromkeys(
        col for f in active_groups for col in config.sel_feats_by_name[f]
    ))
    df = pd.read_csv(config.df_preproc_path, usecols=["BASIC_INFO_ID"] + needed_cols)

    return df["BASIC_INFO_ID"].to_numpy(), df.loc[:, needed_cols].astype(np.float32)


def load_data(
    f_name: str,
    config: Config,
    preproc_cache: tuple[np.ndarray, pd.DataFrame] | None = None,
) -> tuple[list[str], np.ndarray | pd.DataFrame]:
    if f_name == "DTI_FA":
        return get_tbss_processed(
            img_path=config.fa_4d_path,
            subj_dir=config.fa_3d_dir,
            stride=config.downsample,
            mask_path=config.fa_mask_path,
            cache=config.fa_npy_path,
        )

    if f_name == "DTI_MD":
        return get_tbss_processed(
            img_path=config.md_4d_path,
            subj_dir=config.fa_3d_dir,
            stride=config.downsample,
            mask_path=config.fa_mask_path,
            cache=config.md_npy_path,
        )

    if f_name in config.sel_feats_by_name:
        sel_feats = config.sel_feats_by_name[f_name]
        if preproc_cache is not None:
            sid_arr, full_df = preproc_cache
            sub = full_df.loc[:, sel_feats]
            keep = pd.notna(sid_arr) & sub.notna().all(axis=1).to_numpy()
            subj_list = sid_arr[keep].tolist()
            X = sub.loc[keep].reset_index(drop=True)
            custom_print(f"From: {config.df_preproc_path.name} (cached)")
            custom_print(f"{len(subj_list)} participants, {X.shape[1]} features")
            return subj_list, X
        return load_feat_table(config.tbl_paths[f_name], usecols=["BASIC_INFO_ID"] + sel_feats)

    return load_feat_table(config.tbl_paths[f_name])


def run_lv1_models(
    subj_df: pd.DataFrame,
    config: Config,
) -> tuple[list[pd.DataFrame], dict, dict[str, list[str]]]:
    pred_by_feat: list[pd.DataFrame] = []
    summ_by_feat: dict = {}
    lv1_cols_by_feat: dict[str, list[str]] = {}

    src_paths = {**config.tbl_paths, "DTI_FA": config.fa_npy_path, "DTI_MD": config.md_npy_path}
    roster_set = set(subj_df.index)
    preproc_cache = load_preproc_cache(config)
    lv1_kwargs = config.common_train_kwargs(config.model_lv1)

    for f_name in config.feat_types:
        custom_print(f"\nLoading {f_name} data ...")
        subj_list, X = load_data(f_name, config, preproc_cache=preproc_cache)

        keep_mask = np.fromiter((sid in roster_set for sid in subj_list), dtype=bool, count=len(subj_list))
        if not keep_mask.all():
            custom_print(
                f"Filtered out {int((~keep_mask).sum())} participant(s) in {f_name} absent from target table.",
                level="WARNING",
            )
            subj_list = [ sid for sid, ok in zip(subj_list, keep_mask) if ok ]
            X = X.loc[keep_mask].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[keep_mask]

        missing = print_missing(subj_df.index.to_list(), subj_list)

        y = subj_df.loc[subj_list, config.targets].astype(np.float32)
        idx_tr, idx_te = split_indices(subj_df.loc[subj_list, SET])

        verb = "Decompose" if config.decomp else "Train and eval"
        custom_print(f"\n{verb} {config.model_lv1} models on {f_name} features ...")
        outs = train_eval_model(
            X, y, idx_tr, idx_te,
            **lv1_kwargs,
            return_components=config.decomp,
            impute_data=False,
            model_path_template=config.model_paths[f_name],
            model_perf_path=config.perf_paths[f_name],
        )
        y_pred, _, fold_n = outs[:3]
        targ_preds = y_pred.add_suffix(f"_{f_name}")
        # "{block}_PLS{k}" matches eval_feats_imp.py --feat_src cross-decomp
        lv1_out = outs[3].add_prefix(f"{f_name}_") if config.decomp else targ_preds
        lv1_cols_by_feat[f_name] = lv1_out.columns.to_list()

        parts = [pd.DataFrame({SID: subj_list, f"Fold_{f_name}": fold_n})]
        if config.decomp:
            parts.append(targ_preds)  # keep per-block target predictions alongside components
        parts.append(lv1_out)
        pred_by_feat.append(pd.concat(parts, axis=1))

        summ_by_feat[f_name] = {
            "source"       : str(src_paths[f_name]),
            "model_paths"  : str(config.model_paths[f_name]),
            "lv2_columns"  : lv1_cols_by_feat[f_name],
            "n_subjs"      : X.shape[0],
            "n_features"   : X.shape[1],
            "n_train"      : len(idx_tr),
            "n_test"       : len(idx_te),
            "n_missing"    : len(missing),
            "missing_subjs": missing,
            "performance"  : pd.read_csv(config.perf_paths[f_name], index_col="Split").to_dict(orient="index"),
        }

    return pred_by_feat, summ_by_feat, lv1_cols_by_feat


def merge_preds(subj_df: pd.DataFrame, pred_by_feat: list[pd.DataFrame]) -> pd.DataFrame:
    '''
    Join all modality predictions onto `subj_df` in a single indexed concat.
    '''
    aligned = [ df.set_index(SID).reindex(subj_df.index) for df in pred_by_feat ]

    return pd.concat([subj_df, *aligned], axis=1).reset_index()


def add_pyment_results(
    pred_out: pd.DataFrame,
    summ_by_feat: dict,
    lv1_cols_by_feat: dict[str, list[str]],
    config: Config,
) -> tuple[pd.DataFrame, dict, dict[str, list[str]]]:
    f_name = "Pyment"
    pyment_df = pd.read_csv(config.pyment_tbl_path, usecols=["subject", "age"])
    pyment_df = pd.DataFrame({
        SID              : pyment_df["subject"].map(lambda x: f"sub-{x:04d}"),
        f"{AGE}_{f_name}": pyment_df["age"].astype(np.float32),
        f"Fold_{f_name}" : np.int32(-1),  # publicly available pre-trained model
    })

    pred_out = pd.merge(pred_out, pyment_df, on=SID, how="left")
    missing = print_missing(pred_out[SID].to_list(), pyment_df[SID].to_list())

    lv1_cols_by_feat = dict(lv1_cols_by_feat)
    if MIX in config.targets:
        if config.mix_in_years:
            pred_out[f"{MIX}_{f_name}"] = pred_out[f"{AGE}_{f_name}"]
        else:
            tr_idx, _ = split_indices(pred_out[SET])
            age_tr = pred_out[AGE].iloc[tr_idx]
            pred_out[f"{MIX}_{f_name}"] = (pred_out[f"{AGE}_{f_name}"] - age_tr.mean()) / age_tr.std()
        lv1_cols_by_feat[f_name] = [f"{MIX}_{f_name}"]
    else:
        lv1_cols_by_feat[f_name] = [f"{AGE}_{f_name}"]

    valid = pred_out[[AGE, f"{AGE}_{f_name}"]].dropna()
    perf_all = (
        calc_metrics(
            valid[[AGE]].to_numpy(dtype=np.float64),
            valid[[f"{AGE}_{f_name}"]].to_numpy(dtype=np.float64),
            [AGE],
        )
        if len(valid) > 0
        else {f"MAE_{AGE}": np.nan, f"R2_{AGE}": np.nan}
    )

    summ_by_feat[f_name] = {
        "copied_from"  : str(config.pyment_tbl_path),
        "lv2_columns"  : lv1_cols_by_feat[f_name],
        "n_subjs"      : len(valid),
        "n_missing"    : len(missing),
        "missing_subjs": missing,
        "performance"  : {"All": {"N": len(valid), **perf_all}},
    }

    return pred_out, summ_by_feat, lv1_cols_by_feat


def save_pls_feats(pred_out: pd.DataFrame, lv1_cols: list[str], config: Config) -> None:
    '''
    Write PLS components inside `config.lv2_res_dir` so each run keeps its own copy.
    '''
    config.pls_feat_path.parent.mkdir(parents=True, exist_ok=True)
    pred_out.loc[:, [SID] + lv1_cols].to_csv(config.pls_feat_path, index=False)
    custom_print(f"\nSaved: {config.pls_feat_path}")


def run_lv2_model(
    pred_out: pd.DataFrame,
    summ_by_feat: dict,
    lv1_cols: list[str],
    config: Config,
) -> tuple[pd.DataFrame, dict]:
    active_cols = [ c for c in lv1_cols if c in pred_out.columns ]
    idx_tr, idx_te = split_indices(pred_out[SET])
    apply_correction = config.n_folds > 1

    kind = "component" if config.decomp else "prediction"
    custom_print(
        f"\nTrain and eval the final {config.model_lv2} model on the {len(active_cols)} first-level {kind}(s) ..."
    )
    y_pred, y_pred_ac, _ = train_eval_model(
        X=pred_out.loc[:, active_cols].astype(np.float32),
        y=pred_out.loc[:, config.targets].astype(np.float32),
        idx_tr=idx_tr,
        idx_te=idx_te,
        **config.common_train_kwargs(config.model_lv2),
        impute_data=True,
        apply_correction=apply_correction,
        model_path_template=config.lv2_model_path,
        model_perf_path=config.lv2_perf_path,
        calib_param_path=config.lv2_calib_path if apply_correction else None,
    )
    parts = [pred_out, y_pred.add_suffix("_Final")]
    if y_pred_ac is not None:
        parts.append(y_pred_ac.add_suffix("_Final").add_prefix("C-"))
    pred_out = pd.concat(parts, axis=1)

    summ_out = {
        **config.summary_metadata(),
        "n_subjs"    : len(pred_out),
        "n_train"    : len(idx_tr),
        "n_test"     : len(idx_te),
        "predictions": str(config.pred_out_path),
        "feat_types" : summ_by_feat,
        "final"      : {
            "model_paths": str(config.lv2_model_path),
            "features"   : active_cols,
            "performance": pd.read_csv(config.lv2_perf_path, index_col="Split").to_dict(orient="index"),
        },
    }

    return pred_out, summ_out


def main(config: Config) -> None:
    subj_df = load_targets(config)
    custom_print(f"\nNumber of participants: {len(subj_df)}")

    pred_by_feat, summ_by_feat, lv1_cols_by_feat = run_lv1_models(subj_df, config)
    if config.only_run_lv1_models:
        return

    pred_out = merge_preds(subj_df, pred_by_feat)
    pred_out, summ_by_feat, lv1_cols_by_feat = add_pyment_results(
        pred_out, summ_by_feat, lv1_cols_by_feat, config
    )
    lv1_cols = [ col for cols in lv1_cols_by_feat.values() for col in cols ]

    if config.decomp:
        save_pls_feats(pred_out, lv1_cols, config)

    pred_out, summ_out = run_lv2_model(pred_out, summ_by_feat, lv1_cols, config)

    if MIX in config.targets:
        pred_out = restore_from_mix(pred_out)

    config.pred_out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_csv(config.pred_out_path, index=False)
    custom_print(f"\nSaved: {config.pred_out_path}\n")

    config.summ_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.summ_out_path, "w") as f:
        json.dump(to_json_compatible(summ_out), f, allow_nan=False)
    custom_print(f"Saved: {config.summ_out_path}\n")


if __name__ == "__main__":
    config = Config(parse_args())
    log_path = Path(config.log_out_path)

    with tee_output(log_path):
        custom_print(f"\nStart at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main(config)
        custom_print(f"\nFinish at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    custom_print(f"\nDone! logs is saved to: {log_path}\n")