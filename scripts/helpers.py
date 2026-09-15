#!/usr/bin/env python3


import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from utils import custom_print


MODEL_TYPES = ["elasticnet", "lasso", "ridge", "xgboost"]
L1_RATIOS = [.1, .5, .7, .9, .95, .99, 1]
ALPHAS = [1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5]
XGB_PARAM_SPACE = {
    "max_depth"       : lambda t: t.suggest_int("max_depth", 1, 9),
    "learning_rate"   : lambda t: t.suggest_float("learning_rate", 1e-4, 1.0, log=True),
    "n_estimators"    : lambda t: t.suggest_int("n_estimators", 100, 1000),
    "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
    "subsample"       : lambda t: t.suggest_float("subsample", 0.1, 1.0),
    "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.1, 1.0), 
    # "max_bin"         : lambda t: t.suggest_int("max_bin", 32, 128)
}


def print_missing(all_subjs: list[str], data_subjs: list[str]) -> list[str]:
    '''
    Print and return the sorted list of participants 
    that present in `all_subjs` (the reference roster)
    but absent from `data_subjs` (the participants a given modality actually has data for).
    '''
    missing = sorted(set(all_subjs).difference(set(data_subjs)))

    if missing:
        custom_print(f"\ndata of {len(missing)} participant(s) are missing:", level="WARNING")
        for m in missing:
            custom_print(f"\t- {m}")

    return missing


def load_img_data(
    img_path: str | Path,
    print_infos: bool = False,
    get_affine: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    '''
    Load a NIfTI image from `img_path`
    and return its voxel data as a float32 array (with the image's own shape).

    If `print_infos` is set to True, 
    also print the image shape, axis orientation codes, etc.

    If `get_affine` is set to True, 
    also return the 4x4 voxel-to-world affine.
    '''
    
    import nibabel as nib

    def _print_infos(img: nib.nifti1.Nifti1Image):
        hdr = img.header
        custom_print(f"Image shape: {img.shape}")
        custom_print(f"Image orientations: {nib.aff2axcodes(img.affine)}")
        custom_print({
            0: "sform not defined", 
            1: "RAS+ in scanner coordinates", 
            2: "RAS+ aligned to some other scan", 
            3: "RAS+ in Talairach atlas space", 
            4: "RAS+ in MNI atlas space"
        }[int(hdr["sform_code"])])
        custom_print(f'Description: {hdr["descrip"]}')
        custom_print()

    img_path = str(img_path)
    assert os.path.isfile(img_path), f"Image file not exists: {img_path}"

    custom_print(f"Loading image: {img_path}")
    img = nib.load(img_path)

    if print_infos:
        _print_infos(img)

    custom_print("Loading the array data ...")
    img_dat = np.asarray(img.dataobj, dtype=np.float32)  # avoid caching
        # see: https://nipy.org/nibabel/images_and_memory.html#use-the-array-proxy-instead-of-get-fdata

    if get_affine:
        return img_dat, img.affine
    else:
        return img_dat


def get_tbss_processed(
    img_path: str | Path, 
    subj_dir: str | Path,
    stride: int, 
    mask_path: str | Path,
    cache: str | Path
) -> tuple[list[str], np.ndarray]:
    '''
    List the .nii.gz files under `subj_dir` and sort their names (participant IDs), 
    which should be the order TBSS merge per-participant FA volumes.

    Load the 4-D TBSS stack from `img_path`, 
    downsample all three spatial axes by `stride`,
    keep the voxels whose value > 0 in the 3-D mask `mask_path`
    and flatten them per participant.

    Return the participant IDs and the flattened (N, n_vox) array.

    If a processed NPY file (`cache`) exists
    and its sibling JSON file's key matches, reuse it;
    otherwise, rebuilt and save them (cache and key).

    Note: No scaling is applied here; 
    `train_eval_model` fits a `StandardScaler` per fold on its own training subset.
    '''
    def _cache_key():
        '''
        The identity of the result, which includes:
        - The file name and byte size of the source and mask images
        - `stride`
        - The participant IDs, in the order of the volumes
        '''
        def _stat(p):
            p = str(p)
            return {"name": os.path.basename(p), "bytes": os.path.getsize(p)}

        return {
            "img"   : _stat(img_path),
            "mask"  : _stat(mask_path),
            "stride": int(stride),
            "SID"   : subj_list
        }

    cache = Path(cache)
    if cache.suffix != ".npy": 
        cache = cache.with_name(cache.name + ".npy")
        
    key_path = cache.with_suffix(".json")
    subj_list = [ fp.name.split(".")[0] for fp in sorted(Path(subj_dir).glob("*.nii.gz")) ]
    has_sources = bool(subj_list) and os.path.isfile(img_path) and os.path.isfile(mask_path)
    key = _cache_key()

    if cache.exists():
        try:
            cached_key = json.loads(key_path.read_text())
        except (OSError, ValueError):
            cached_key = None

        if not has_sources:
            msg1 = "\nThe source images are not found, "
            msg2 = f"\nRebuild {cache.name} where the source images are.\n"
            assert cached_key, (msg1 + "nor a readable cache key exists." + msg2)
            assert "SID" in cached_key, (msg1 + f"and {key_path.name} holds no participant IDs." + msg2)
            assert cached_key.get("stride") == int(stride), (msg1 + f"yet the cache is built with a stride size of {cached_key.get("stride")}." + msg2)
            custom_print(f"Loaded from cache, without the source images to check it against: {cache}", level="WARNING")
            return cached_key["SID"], np.load(cache, mmap_mode="r")

        if cached_key == key:
            custom_print(f"Loaded from cache: {cache}")
            return subj_list, np.load(cache, mmap_mode="r")

        stale = (
            "no readable cache key" if cached_key is None 
            else ", ".join(
                "participant IDs" if k == "SID" else f"{k}: {cached_key.get(k)!r} -> {v!r}" 
                for k, v in key.items() if cached_key.get(k) != v
            )
        )
        custom_print(f"Stale cache, recomputing ({stale}): {cache}", level="WARNING")

    assert has_sources, (
        f"\nCannot build {cache.name}; the source images are not found "
        f"(stack: {img_path}, mask: {mask_path}, per-participant images: {subj_dir})\n"
    )
    img_dat = load_img_data(img_path)
    N = img_dat.shape[-1]
    assert N == len(subj_list), f"Mismatch between volume ({N}) and globbed ({len(subj_list)}) subject count."
    
    img_dat = img_dat[::stride, ::stride, ::stride, :]
    img_dat = np.moveaxis(img_dat, -1, 0)  # (N, X, Y, Z)

    mask_dat = load_img_data(mask_path)
    mask_dat = mask_dat[::stride, ::stride, ::stride]
    bin_mask = mask_dat > 0

    img_flat = img_dat[:, bin_mask].copy()  # (N, n_vox)

    if cache.exists():  # keep the old cache and its key
        old = cache.with_name(cache.stem + "_old")
        while old.with_suffix(".npy").exists():
            old = old.with_name(old.name + "+")
        os.replace(src=cache, dst=old.with_suffix(".npy"))
        if key_path.exists():
            os.replace(key_path, old.with_suffix(".json"))

    np.save(cache, img_flat)
    key_path.write_text(json.dumps(key, indent=2))
    custom_print(f"Saved cache ({img_flat.nbytes / 1e9:.2f} GB): {cache}")

    return subj_list, img_flat


def load_feat_table(
    tbl_path: str | Path,
    prefer_npz: bool = True,
    usecols: list[str] | None = None,
    id_col: str | None = None
) -> tuple[list[str], pd.DataFrame]:
    '''
    Load the feature matrix and the participant IDs which it belongs 
    from `tbl_path` (can be a CSV file or a NPZ file).
    
    If `prefer_npz` is set to True and the `tbl_path` points to a CSV file, 
    try to load from its sibling NPZ file (".npz" of the same stem).

    Although the layout of CSV / NPZ files differ slightly, 
    both should contain an ID column + named feature columns 
    (see also `make_df_*.py` scripts):
    - CSV: Participant IDs are stored along with the feature value, 
           with a header row storing their names.
           The ID column is named `id_col`, or the first column when `id_col` is omitted.
    - NPZ: Participant IDs are stored in the "SID" block, 
           feature values are stored in the "X" block, and
           feature names are stored in the "feats" block.

    One can use `usecols` to select features, 
    but the column order always follows the original table.

    Notes: 
    - If X is loaded from a NPZ file without a "feats" (p.s., cannot use `usecols`), 
      positional placeholders ("col-0", ...) will be generated.
    - Rows are dropped when the ID is missing or *any* selected feature is NaN.
    '''
    def _load_from_npz():
        with np.load(tbl_path, allow_pickle=True) as dat:
            if usecols is not None:
                assert "feats" in dat, "'usecols' needs the feature names, but the NPZ file has no 'feats' key"
                sel = np.isin(dat["feats"], usecols)  # boolean mask
                X = dat["X"][:, sel].astype(np.float32)
                feat_list = dat["feats"][sel].tolist()
                return (
                    dat["SID"], 
                    pd.DataFrame(X, columns=feat_list)
                )
            else:
                if "feats" not in dat:
                    custom_print(f"{tbl_path.name} has no 'feats' key; naming its columns positionally ('col-0', ...).")
                    feat_list = [ f"col-{i}" for i in range(dat["X"].shape[1]) ]
                else:
                    feat_list = dat["feats"].tolist()
                return (
                    dat["SID"], 
                    pd.DataFrame(dat["X"].astype(np.float32), columns=feat_list)
                )

    def _load_from_csv():
        nonlocal id_col
        df = pd.read_csv(tbl_path, usecols=usecols)
        id_col = df.columns.tolist()[0] if id_col is None else id_col
        feat_list = [ c for c in df.columns if c != id_col ]
        return (
            df[id_col].to_numpy(),  # convert into a np.array to use fancy indexing 
            df.loc[:, feat_list].astype(np.float32)
        )

    tbl_path = Path(tbl_path)

    if tbl_path.suffix.lower() not in [".npz", ".csv"]:
        raise ValueError(f"Unsupported table format: '{tbl_path.suffix}' ({tbl_path})")

    if prefer_npz:
        tbl_path = tbl_path.with_suffix(".npz")
    
    if tbl_path.suffix.lower() == ".npz" and tbl_path.exists():
        subj_arr, X = _load_from_npz()
    else:
        tbl_path = tbl_path.with_suffix(".csv")
        subj_arr, X = _load_from_csv()

    keep = X.notna().all(axis=1).to_numpy()
    subj_list = subj_arr[keep].tolist() 
    X = X[keep].reset_index(drop=True)

    custom_print(f"From: {tbl_path.name}")
    custom_print(f"{len(subj_list)} participants, {X.shape[1]} features")

    return subj_list, X


def train_eval_model(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame,
    idx_tr: np.ndarray, 
    idx_te: np.ndarray, 
    model_type: str, 
    seed: int = 42, 
    seed_inner: int = 0, 
    n_folds: int = 5, 
    l1_ratios: list[float] = L1_RATIOS, 
    alphas: list[float] = ALPHAS, 
    max_iter: int = 10000, 
    xgb_params: dict = {}, 
    opt_trials: int = 100, 
    n_jobs: int = -1, 
    xgb_device: str = "cpu", 
    verbose: int = 1, 
    impute_data: bool = False,
    perf_metrix: str = "MAE",
    apply_correction: bool = False,
    model_path_template: str | Path | None = None,
    model_perf_path: str | Path | None = None,
    calib_param_path: str | Path | None = None,
    overwrite: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray]:
    '''
    Fit one or more [optional imputer -> scaler -> regressor] pipeline(s) to 
    produce train/test predictions, their age-corrected counterparts (optional), 
    plus the indices of folds to which they belong.

    Fitted models, their performance metrics, and age-correction parameters 
    can optionally be cached to disk.

    If multiple targets are present in `y`, 
    they are scaled to a common range (see `TargetScaler`) and fit jointly.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
        Input feature matrix.

    y : np.ndarray or pd.DataFrame of shape (n_samples,) or (n_samples, n_targets)
        Target vector or table.

    idx_tr : np.ndarray of shape (n_train,)
        Integer row indices of `X`/`y` assigned to the training/cross-validation set.

    idx_te : np.ndarray of shape (n_test,)
        Integer row indices of `X`/`y` assigned to the external test set.

    model_type : str
        Type of regression algorithm to use. Must be one of `MODEL_TYPES` 
        (e.g., "elasticnet", "lasso", "ridge", "xgboost").

    seed : int, default=42
        Random seed for the outer cross-validation splits on `idx_tr`.

    seed_inner : int, default=0
        Random seed for the inner cross-validation (hyperparameter tuning).

    n_folds : int, default=5
        Number of outer CV folds. If `n_folds <= 1`, the entire training set is 
        used to fit a single model without out-of-fold validation.

    l1_ratios : list of float, default=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        L1 penalty mixing parameter grid for ElasticNet models.

    alphas : list of float, default=[1e-1, ..., 1e5]
        Regularization strength grid for linear models.

    max_iter : int, default=10000
        Maximum number of iterations for the linear solvers.

    xgb_params : dict, optional
        Custom hyperparameters for XGBoost models.

    opt_trials : int, default=100
        Number of Optuna trials spent on searching the best XGBoost hyperparameters

    n_jobs : int, default=-1
        Number of CPU cores used for parallel execution (-1 uses all available).

    xgb_device : {"cpu", "cuda"}, default="cpu"
        Device on which the XGBoost models are trained. 
        With "cuda", the folds of each Optuna trial are fitted one at a time, 
        and a trial or a fit that runs out of GPU memory is repeated on the CPU.

    verbose : int, default=1
        Verbosity level of execution logs.

    impute_data : bool, default=False
        If True, prepends an imputation step at the beginning of the pipeline.

    perf_metrix : {"MAE", "R2"}, default="MAE"
        Metric used to select the "best" outer fold model for evaluating `idx_te`.
        Averaged across targets if multi-target.

    apply_correction : bool, default=False
        If True, fits an age-bias calibration per-fold/per-target and returns 
        bias-corrected predictions. Requires `n_folds > 1`.

    model_path_template : str or Path, optional
        File path pattern to save/load fitted pipelines. Must contain a single `"{}"` 
        placeholder (formatted as `"fold-{i}"` or `"all"`) and end with `".joblib"`.
        If cached models exist and match target names, they are loaded unless 
        `overwrite=True`.

    model_perf_path : str or Path, optional
        Path to save model evaluation metrics as a long-format CSV. If omitted,
        metrics are not written to disk.

    calib_param_path : str or Path, optional
        Path to save age-calibration parameters (`const_*`, `slope_*`) as a CSV.
        Only used if `apply_correction=True`.

    overwrite : bool, default=False
        If True, re-trains and overwrites cached model files at `model_path_template`.
        (Metric and calibration CSVs are always overwritten regardless).

    Returns
    -------
    y_pred : pd.DataFrame
        Raw predictions. For `idx_tr`, predictions are out-of-fold (if `n_folds > 1`)
        or in-sample (if `n_folds <= 1`). For `idx_te`, predictions are generated 
        by the best-performing fold model.
        
    y_pred_ac : pd.DataFrame | None
        Age-corrected predictions if `apply_correction=True`, otherwise `None`.

    fold_n : np.ndarray
        Outer fold index for each sample:
        - `n_folds - 1`: Outer validation fold index for training rows.
        - `0`: For all training rows if `n_folds <= 1`.
        - `-1`: For testing rows (`idx_te`) and unselected samples.

    Raises
    ------
    AssertionError
    - Dimensions of `X` and `y` do not match.
    - `y` carries no target, a non-numeric one, or NaN.
    - `idx_tr` / `idx_te` are not integer indices, are empty, are out of bounds, or overlap.
    - `model_type` or `perf_metrix` is invalid.
    - `apply_correction=True` when `n_folds <= 1`.
    - `calib_param_path` is specified but `apply_correction=False`.
    - `model_path_template` violates filename conventions.
    '''

    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, MultiTaskElasticNetCV, MultiTaskLassoCV
    from sklearn.model_selection import KFold

    if model_type == "xgboost":
        import optuna
        # import optunahub
        from xgboost import XGBRegressor
        from sklearn.metrics import make_scorer, mean_absolute_error
        from sklearn.model_selection import KFold, cross_val_score

    def _validate_inputs():
        assert X.shape[0] == len(y), f"\nMismatch between length of X ({X.shape[0]}) and y ({len(y)})\n"
        assert n_targets > 0, "\ny carries no target\n"
        assert Y.dtypes.map(pd.api.types.is_numeric_dtype).all(), "\ny carries non-numeric target(s)\n"
        assert not Y.isna().any(axis=None), "\ny carries NaN target(s)\n"
        assert idx_tr.dtype.kind in "iu" and idx_te.dtype.kind in "iu", f"\nidx_tr/idx_te must be integer indices, got {idx_tr.dtype}/{idx_te.dtype}\n"

        overlap = np.intersect1d(idx_tr, idx_te)
        assert overlap.size == 0, f"\n{overlap.size} overlap(s) between training and testing indices\n"

        for nm, idx in [("idx_tr", idx_tr), ("idx_te", idx_te)]:
            assert idx.size > 0, f"\n{nm} is empty\n"
            assert idx.min() >= 0 and idx.max() < len(y), f"\n{nm} out of bounds: [{idx.min()}, {idx.max()}] vs len(y)={len(y)}\n"
        
        assert model_type in MODEL_TYPES, f"\nModel type '{model_type}' is undefined.\n"
        assert perf_metrix in {"MAE", "R2"}, f"\nPerformance metrix '{perf_metrix}' is undefined.\n"
        assert xgb_device in {"cpu", "cuda"}, f"\nDevice '{xgb_device}' is invalid.\n"

        assert not (apply_correction and n_folds <= 1), (
            "\n'apply_correction' fits one slope per fold on that fold's held-out rows, "
            f"but n_folds is {n_folds}; there are no held-out rows to fit on.\n"
        )

        if model_path_template:
            assert "{}" in model_path_template, "\n'model_path_template' should have '{}', got:\n" + model_path_template
            assert model_path_template.endswith(".joblib"), f"\n'model_path_template' should end with .joblib, got:\n{model_path_template}"
            _dir_1 = os.path.dirname(model_path_template.format("x"))
            if _dir_1:
                os.makedirs(_dir_1, exist_ok=True)
        else:
            custom_print("\n'model_path_template' is not defined. No trained model will be saved.\n", level="WARNING")

        if model_perf_path:
            assert model_perf_path.endswith(".csv"), f"\n'model_perf_path' should end with .csv, got:\n{model_perf_path}"
            _dir_2 = os.path.dirname(model_perf_path)
            if _dir_2:
                os.makedirs(_dir_2, exist_ok=True)
        else:
            custom_print("\n'model_perf_path' is not defined. Model performances will not be calculated and saved.\n", level="WARNING")

        if calib_param_path:
            assert apply_correction, "\n'calib_param_path' is given but 'apply_correction' is False, so there is nothing to record.\n"
            assert calib_param_path.endswith(".csv"), f"\n'calib_param_path' should end with .csv, got:\n{calib_param_path}"
            _dir_3 = os.path.dirname(calib_param_path)
            if _dir_3:
                os.makedirs(_dir_3, exist_ok=True)        

    def _use_cached_model(model_path: str) -> bool:
        return bool(model_path_template) and os.path.isfile(model_path) and not overwrite

    def _get_features(idx: np.ndarray) -> np.ndarray | pd.DataFrame:
        # The parameter X is read directly from the local variable
        return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]

    def _get_targets(idx: np.ndarray) -> np.ndarray:
        return y_arr[idx] if n_targets > 1 else y_arr[idx, 0]

    def _get_target_scales(idx: np.ndarray) -> np.ndarray:
        stds = np.nanstd(y_arr[idx], axis=0)
        stds[stds == 0] = 1.
        return stds / stds[0]

    def _init_pipeline(temp_xgb_params: dict | None = None, device: str = xgb_device):
        multi = n_targets > 1
        _kf = KFold(n_splits=5, shuffle=True, random_state=seed_inner)

        regressor = {
            "elasticnet": lambda: (MultiTaskElasticNetCV if multi else ElasticNetCV)(
                l1_ratio=l1_ratios, 
                alphas=alphas, 
                cv=_kf, 
                max_iter=max_iter, 
                n_jobs=n_jobs, 
                verbose=verbose
            ), 
            "lasso": lambda: (MultiTaskLassoCV if multi else LassoCV)(
                alphas=alphas, 
                cv=_kf, 
                max_iter=max_iter, 
                n_jobs=n_jobs, 
                verbose=verbose
            ), 
            "ridge": lambda: RidgeCV(  # natively supports multi-output
                alphas=alphas, 
                cv=_kf
            ), 
            "xgboost": lambda: XGBRegressor(
                **xgb_params, 
                **(temp_xgb_params or best_xgb_params), 
                multi_strategy="multi_output_tree" if multi else "one_output_per_tree", 
                random_state=seed_inner, 
                device=device, 
                n_jobs=n_jobs, 
                verbosity=verbose
            )
        }[model_type]()

        if multi:
            regressor = TransformedTargetRegressor(
                regressor=regressor, transformer=TargetScaler()
            )

        steps = [("scaler", StandardScaler()), ("model", regressor)]

        if impute_data:
            steps.insert(0, ("imputer", SimpleImputer(strategy="median")))

        return Pipeline(steps=steps)

    def _eval_xgb_params(trial, idx: np.ndarray, device: str = xgb_device):
        params = {
            k: fun(trial) for k, fun in XGB_PARAM_SPACE.items() 
            if k not in xgb_params.keys()
        }
        try:
            neg_mae_scores = cross_val_score(
                estimator=_init_pipeline(temp_xgb_params=params, device=device), 
                X=_get_features(idx), 
                y=_get_targets(idx), 
                cv=KFold(n_splits=5, shuffle=True, random_state=seed_inner), 
                scoring=(
                    make_scorer(  # weight every target by its own std, as TargetScaler does
                        mean_absolute_error, greater_is_better=False, 
                        multioutput=1 / _get_target_scales(idx)
                    ) if n_targets > 1 else "neg_mean_absolute_error"
                ), 
                n_jobs=1 if (device == "cuda") else n_jobs, 
                error_score="raise" if (device == "cuda") else np.nan,  # stop at the first fold that runs out of memory
                verbose=verbose
            )
        except Exception as e:
            if (device == "cuda") or ("out of memory" in str(e)):
                custom_print(f"\nTrial {trial.number} runs out of GPU memory; evaluating it on the CPU instead.", level="WARNING")
                return _eval_xgb_params(trial, idx, device="cpu")
            else:
                raise

        return -1 * np.mean(neg_mae_scores)

    def _optimize_xgb_params(idx: np.ndarray):
        # try:
        #     module = optunahub.load_module(package="samplers/auto_sampler")
        #     sampler = module.AutoSampler(seed=seed_inner)
        # except Exception as e:
        #     reason = str(e).splitlines()[0] if str(e) else type(e).__name__
        #     custom_print(f"\nFalling back on TPESampler, AutoSampler is unavailable: {reason}", level="WARNING")
        #     sampler = optuna.samplers.TPESampler(seed=seed_inner)
        sampler = optuna.samplers.TPESampler(seed=seed_inner)

        optuna.logging.set_verbosity(optuna.logging.INFO if verbose > 0 else optuna.logging.WARNING)
        custom_print(f"\nSearching the XGBoost hyperparameters over {opt_trials} trial(s) ...")
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(lambda trial: _eval_xgb_params(trial, idx), n_trials=opt_trials)
        custom_print("\nParameter optimization is completed :-)")
        custom_print(f"Cross-validated MAE of the best trial: {study.best_value:.3f}")
        for k, v in study.best_params.items():
            custom_print(f"\t{k}: {v}")

        return study.best_params

    def _fit_or_load(suffix: str, pipeline, idx: np.ndarray):
        model_path = model_path_template.format(suffix) if model_path_template else None

        if _use_cached_model(model_path):
            pipeline = joblib.load(model_path)            
            cached_targets = list(getattr(pipeline, "target_names_in_", []))
            assert cached_targets == targets, (
                f"\n{os.path.basename(model_path)} was fit on {cached_targets or 'unrecorded target(s)'}, not on {targets}. "
                "Refit it by setting 'overwrite' to True.\n"
            )
            custom_print(f"Loaded pre-trained: {model_path}")

        else:
            try:
                pipeline.fit(_get_features(idx), _get_targets(idx))
            except Exception as e:
                if (model_type == "xgboost") and (xgb_device == "cuda") and ("out of memory" in str(e)):
                    custom_print(f"\nThe '{suffix}' model runs out of GPU memory; fitting it on the CPU instead.", level="WARNING")
                    pipeline = _init_pipeline(device="cpu")
                else:
                    raise

            pipeline.target_names_in_ = targets  # self-define property

            if model_path:
                joblib.dump(pipeline, model_path)
                custom_print(f"\nSaved: {model_path}\n")

        return pipeline

    def _predict(pipeline, idx: np.ndarray) -> np.ndarray:
        # returns in a 2-D shape even if n_targets == 1, whose default is (n,) instead of (n, 1) 
        return np.asarray(
            pipeline.predict(_get_features(idx))
        ).reshape(len(idx), n_targets)

    def _calc_model_perf(idx: np.ndarray):
        perf = {}

        for i, target in enumerate(targets):
            err = y_arr[idx, i] - y_pred[idx, i]
            mae = np.mean(np.abs(err))
            r2 = 1 - np.sum(err ** 2) / np.sum((y_arr[idx, i] - y_arr[idx, i].mean()) ** 2)
            perf.update({
                f"MAE_{target}": mae, 
                f"R2_{target}": r2
            })

        return perf

    def _fit_calibrator(idx: np.ndarray) -> dict[str, float]:
        '''
        For every target, 
        fit a simple linear regression of predictions on the true values,
        which will then be used to reverse-transform the predictions
        using the resulting slope and intercept.
        (i.e., Apply age prediction correction with Cole's method).

        Called on one fold's held-out rows.
        '''
        all_coefs = {}

        for i, target in enumerate(targets):
            coefs = np.polyfit(y_arr[idx, i], y_pred[idx, i], 1)
            all_coefs.update({
                f"const_{target}": coefs[1], 
                f"slope_{target}": coefs[0]
            })

        return all_coefs

    def _apply_calibrator(idx: np.ndarray, coefs: dict[str, float]):
        for i, target in enumerate(targets):
            y_pred_ac[idx, i] = (
                (y_pred[idx, i] - coefs[f"const_{target}"]) / coefs[f"slope_{target}"]
            )

    ## ------------------------------------------------------------------------------

    model_path_template = None if model_path_template is None else str(model_path_template)
    model_perf_path = None if model_perf_path is None else str(model_perf_path)
    calib_param_path = None if calib_param_path is None else str(calib_param_path)

    Y = y if isinstance(y, pd.DataFrame) else pd.DataFrame(np.asarray(y).reshape(len(y), -1)).add_prefix("y")
    Y = Y.reset_index(drop=True)  # the indices below are positional
    targets = Y.columns.to_list()
    n_targets = len(targets)
    y_arr = Y.to_numpy()

    _validate_inputs()

    n_samples = len(y_arr)
    y_pred = np.full((n_samples, n_targets), np.nan, dtype=np.float32)
    y_pred_ac = np.full_like(y_pred, np.nan) if apply_correction else None
    fold_n = np.full(n_samples, -1, dtype=np.int8)
    perfs = []
    best_score = np.inf if perf_metrix == "MAE" else -np.inf

    best_xgb_params = {}
    if model_type == "xgboost":
        cuda_is_available = bool(XGBRegressor(tree_method="hist", device="cuda").fit([[0]], [0]))
        xgb_device = "cpu" if not cuda_is_available else xgb_device
        
        skip_tune = False
        if not (XGB_PARAM_SPACE.keys() - xgb_params.keys()):
            custom_print("\n'xgb_params' fixes every tunable hyperparameter; no search is needed.\n")
            skip_tune = True
        if (not skip_tune) and bool(model_path_template):
            suffixes = [ f"fold-{k}" for k in range(n_folds) ] if n_folds > 1 else [ "all" ]
            model_paths = [ model_path_template.format(s) for s in suffixes ]
            skip_tune = all( _use_cached_model(p) for p in model_paths )
            if skip_tune:
                custom_print("\nEvery model is already trained; the hyperparameter search is skipped.\n")
        if not skip_tune:
            assert opt_trials > 0, "To perform hyperparameter tuning, `opt_trials` must be greater than zero."
            best_xgb_params = _optimize_xgb_params(idx_tr)

    if n_folds > 1:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        pipes, calibs = {}, {}
        for k, (tr, va) in enumerate(kf.split(idx_tr)):
            pipes[k] = _fit_or_load(f"fold-{k}", _init_pipeline(), idx_tr[tr])
            y_pred[idx_tr[va]] = _predict(pipes[k], idx_tr[va])
            fold_n[idx_tr[va]] = k

            if apply_correction:
                calibs[k] = _fit_calibrator(idx_tr[va])
                _apply_calibrator(idx_tr[va], calibs[k])
           
            perf_va = _calc_model_perf(idx_tr[va])
            score = np.mean([ perf_va[f"{perf_metrix}_{t}"] for t in targets ])

            improved = (score < best_score) if perf_metrix == "MAE" else (score > best_score)
            if improved:
                best_score = score
                best_fold = k

            if model_perf_path:
                perfs.append(
                    {"Split": f"Val_fold-{k}", "N": len(idx_tr[va]), **perf_va}
                )

        best_pipe = pipes[best_fold]
        y_pred[idx_te] = _predict(best_pipe, idx_te)

        if apply_correction:
            _apply_calibrator(idx_te, calibs[best_fold])

        if model_perf_path:
            fold_perfs = pd.DataFrame(perfs).drop(columns=["Split", "N"])
            perf_tr = _calc_model_perf(idx_tr)
            perf_te = _calc_model_perf(idx_te)
            perfs += [
                {"Split": "Val_mean"  , "N": None       , **fold_perfs.mean().to_dict()},
                {"Split": "Val_SD"    , "N": None       , **fold_perfs.std(ddof=1).to_dict()},
                {"Split": "Val_pooled", "N": len(idx_tr), **perf_tr}, 
                {"Split": "Test"      , "N": len(idx_te), **perf_te}
            ]

    else: # both predict by the model train on entire training set
        pipe = _fit_or_load("all", _init_pipeline(), idx_tr)
        y_pred[idx_tr] = _predict(pipe, idx_tr)
        fold_n[idx_tr] = 0

        y_pred[idx_te] = _predict(pipe, idx_te)

        if model_perf_path:
            perf_tr = _calc_model_perf(idx_tr)
            perf_te = _calc_model_perf(idx_te)
            perfs += [
                {"Split": "Train", "N": len(idx_tr), **perf_tr},
                {"Split": "Test" , "N": len(idx_te), **perf_te}
            ]
        
    if model_perf_path:
        perfs = pd.DataFrame(perfs)
        perfs["N"] = perfs["N"].astype("Int64")  # nullable int
        perfs.to_csv(model_perf_path, index=False)
        custom_print(f"\nSaved: {model_perf_path}")
        custom_print("Test MAE: " + ", ".join( f"{t} = {perf_te[f'MAE_{t}']:.2f}" for t in targets ))

    if calib_param_path:
        calib_df = pd.DataFrame.from_dict(calibs, orient="index")
        calib_df.index.name = "Fold"
        calib_df.to_csv(calib_param_path)
        custom_print(f"\nSaved: {calib_param_path}")

    return (
        pd.DataFrame(y_pred, columns=targets), 
        pd.DataFrame(y_pred_ac, columns=targets) if apply_correction else None, 
        fold_n
    )


class TargetScaler(BaseEstimator, TransformerMixin):
    '''
    Scale multi-output targets relative to the first column.

    Notes
    -----
    - NaNs are ignored when estimating mean and std, 
      but are preserved in `transform` and `inverse_transform`.
    - `y` argument exists only for API compatibility.
    '''
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        '''
        Estimate per-column nan-aware mean 
        and a scale derived from nan-aware standard deviations
        (expressed in units of column 0's std). 
        '''
        X = np.asarray(X, dtype=np.float64)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.  # prevent division by zero for constant column(s)

        self.center_ = np.nanmean(X, axis=0)
        self.scale_ = std / std[0]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.center_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.center_

