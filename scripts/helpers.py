#!/usr/bin/env python3


import os
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from utils import custom_print


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
    N: int,
    stride: int, 
    mask_path: str | Path,
    cache: str | Path
) -> np.ndarray:
    '''
    Load the 4-D TBSS stack from `img_path`, 
    check its last axis against the expected participant count `N`, 
    downsample all three spatial axes by `stride`,
    keep the voxels whose value > 0 in the 3-D mask `mask_path`
    and flatten them per participant.

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
        - `stride` and `N`
        '''
        def _stat(p):
            p = str(p)
            return {"name": os.path.basename(p), "bytes": os.path.getsize(p)}

        return {
            "img"   : _stat(img_path),
            "mask"  : _stat(mask_path),
            "stride": int(stride),
            "N"     : int(N)
        }

    cache = Path(cache)
    if cache.suffix != ".npy": 
        cache = cache.with_name(cache.name + ".npy")
        
    key_path = cache.with_suffix(".json")
    key = _cache_key()

    if cache.exists():
        try:
            cached_key = json.loads(key_path.read_text())
        except (OSError, ValueError):
            cached_key = None

        if cached_key == key:
            custom_print(f"Loaded from cache: {cache}")
            return np.load(cache, mmap_mode="r")

        stale = (
            "no readable cache key" if cached_key is None 
            else ", ".join( f"{k}: {cached_key.get(k)!r} -> {v!r}" for k, v in key.items() if cached_key.get(k) != v )
        )
        custom_print(f"Stale cache, recomputing ({stale}): {cache}", level="WARNING")

    img_dat = load_img_data(img_path)
    N_v = img_dat.shape[-1]
    assert N_v == N, f"Mismatch between volume ({N_v}) and globbed ({N}) subject count."
    
    img_dat = img_dat[::stride, ::stride, ::stride, :]
    img_dat = np.moveaxis(img_dat, -1, 0)  # (N, X, Y, Z)

    mask_dat = load_img_data(mask_path)
    mask_dat = mask_dat[::stride, ::stride, ::stride]
    bin_mask = mask_dat > 0

    img_flat = img_dat[:, bin_mask].copy()  # (N, n_vox)

    np.save(cache, img_flat)
    key_path.write_text(json.dumps(key, indent=2))
    custom_print(f"Saved cache ({img_flat.nbytes / 1e9:.2f} GB): {cache}")

    return img_flat


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
    y: np.ndarray,
    idx_tr: np.ndarray, 
    idx_te: np.ndarray, 
    model_type: str, 
    seed: int = 42, 
    seed_inner: int = 0, 
    n_folds: int = 5, 
    l1_ratios: list[float] = [.1, .5, .7, .9, .95, .99, 1], 
    alphas: list[float] = [1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5], 
    max_iter: int = 10000, 
    n_jobs: int = -1, 
    verbose: int = 1, 
    impute_data: bool = False,
    perf_metrix: str = "MAE",
    apply_correction: bool = False,
    model_path_template: str | Path = None,
    model_perf_path: str | Path = None,
    calib_param_path: str | Path = None,
    overwrite: bool = False
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    '''
    Fit one or many [scaler -> CV regressor] pipeline(s) and return training and testing
    predictions, their age-corrected counterpart, and fold indices.

    Fold indices are `n_folds`- 1 at which each training sample was held out; 
    -1 for testing and unused samples. 0 for training samples if n_folds <= 1.

    Pipeline(s) begin with an imputer if `impute_data` is set to True.

    The type of CV regressor is set by `model_type`, 
    which can be "elasticnet", "lasso", or "ridge".
    (CV is for regularization strength, always 5 folds, seeded by `seed_inner`).

    For training set (data rows indexed by `idx_tr`), if `n_folds` > 1, 
    predictions are generated over `n_folds` folds (seeded by `seed`)
    from models trained on out-of-fold data; 
    otherwise, predictions are generated from the model trained on itself.
    
    For testing set (data rows indexed by `idx_te`),
    predictions are generated from the model trained on best fold,
    which is determined by `perf_metrix` ("MAE" or "R2").

    If `model_path_template` is given, 
    fitted pipelines are cached to disk (parent directory is created if needed),
    or an existing file is loaded (unless `overwrite` is set to True).
    If omitted, models are neither loaded nor saved.
    It must contain a single "{}" placeholder (will be filled with "fold-*" or "all")
    and end with ".joblib".

    If `model_perf_path` is given, 
    a CSV file will be created to write performance metrics in long format,
    with columns `Split`, `N`, `MAE`, `R2` and one row per split:
    - "Val_fold-{0 ... n-1}": each outer fold's held-out samples
    - "Val_mean" / "Val_SD" : mean and sample SD (ddof=1) across those folds
    - "Val_pooled"          : all out-of-fold predictions on `idx_tr` scored at once
    - "Test"                : predictions on `idx_te`
    In the `n_folds <= 1` mode, the rows are instead "Train" and "Test".
    Always (re)written when given, regardless of `overwrite`.
    If omitted, metrics are neither computed nor saved.

    If `apply_correction` is set to True, 
    the second returned array holds the age-corrected predictions 
    generated through a per-fold calibration (see `_fit_calibrator`); 
    it is `None` otherwise. 

    If `calib_param_path` is given when `apply_correction` is True,
    a CSV file will be created to record the fitted parameters, one row per fold.
    Always (re)written when given, regardless of `overwrite`.
    If omitted, the parameters are applied but not saved.

    Raises
    ------
    AssertionError
    - `X` / `y` lengths mismatch
    - `y` is not 1-D
    - `idx_tr` / `idx_te` are not integer indices, are empty, are out of bounds, or overlap
    - `model_type` or `perf_metrix` is unknown
    - `apply_correction` is asked for with `n_folds` <= 1
    - `calib_param_path` is given without `apply_correction`
    - the output paths violate the format constraints above
    '''

    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.model_selection import KFold

    def _validate_inputs():
        assert X.shape[0] == len(y), f"\nMismatch between length of X ({X.shape[0]}) and y ({len(y)})\n"
        assert y.ndim == 1, f"\ny must be 1-D, got shape {y.shape}\n"
        assert idx_tr.dtype.kind in "iu" and idx_te.dtype.kind in "iu", f"\nidx_tr/idx_te must be integer indices, got {idx_tr.dtype}/{idx_te.dtype}\n"

        overlap = np.intersect1d(idx_tr, idx_te)
        assert overlap.size == 0, f"\n{overlap.size} overlap(s) between training and testing indices\n"

        for nm, idx in [("idx_tr", idx_tr), ("idx_te", idx_te)]:
            assert idx.size > 0, f"\n{nm} is empty\n"
            assert idx.min() >= 0 and idx.max() < len(y), f"\n{nm} out of bounds: [{idx.min()}, {idx.max()}] vs len(y)={len(y)}\n"
        
        assert model_type in {"elasticnet", "lasso", "ridge"}, f"\nModel type '{model_type}' is undefined.\n"
        assert perf_metrix in {"MAE", "R2"}, f"\nPerformance metrix '{perf_metrix}' is undefined.\n"

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

    def _init_model():
        _kf = KFold(n_splits=5, shuffle=True, random_state=seed_inner)
        model_name_2_obj = {
            "elasticnet": ElasticNetCV(
                l1_ratio=l1_ratios, 
                alphas=alphas, 
                cv=_kf, 
                max_iter=max_iter, 
                n_jobs=n_jobs, 
                verbose=verbose
            ), 
            "lasso": LassoCV(
                alphas=alphas, 
                cv=_kf, 
                max_iter=max_iter, 
                n_jobs=n_jobs, 
                verbose=verbose
            ), 
            "ridge": RidgeCV(
                alphas=alphas, 
                cv=_kf
            )
        }

        if model_type not in model_name_2_obj.keys():
            raise ValueError(f"Unknown model_type: {model_type}")

        if impute_data:
            return Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model_name_2_obj[model_type])
            ])
        else:
            return Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("model", model_name_2_obj[model_type])
            ])

    def _get_rows(idx: np.ndarray) -> np.ndarray | pd.DataFrame:
        # The parameter X is read directly from the local variable
        return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]

    def _fit_or_load(suffix: str, model, X_fit: np.ndarray | pd.DataFrame, y_fit: np.ndarray):
        model_path = model_path_template.format(suffix) if model_path_template else None

        if model_path and os.path.isfile(model_path) and not overwrite:
            custom_print(f"Loaded pre-trained: {model_path}")
            return joblib.load(model_path)

        model.fit(X_fit, y_fit)

        if model_path:
            joblib.dump(model, model_path)
            custom_print(f"\nSaved: {model_path}\n")

        return model

    def _calc_model_perf(y_true: np.ndarray, y_pred: np.ndarray):
        err = y_true - y_pred
        mae = np.mean(np.abs(err))
        r2 = 1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        return {"MAE": mae, "R2": r2}

    def _fit_calibrator(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        '''
        Fit a simple linear regression of `y_pred` on `y_true`,
        which will then be used to reverse-transform the predictions
        using the resulting slope and intercept.
        (i.e., Apply age prediction correction with Cole's method).

        Called on one fold's held-out rows.
        '''
        coefs = np.polyfit(y_true, y_pred, 1)
        return {"intercept": coefs[1], "slope": coefs[0]}

    ## ------------------------------------------------------------------------------

    model_path_template = None if model_path_template is None else str(model_path_template)
    model_perf_path = None if model_perf_path is None else str(model_perf_path)
    calib_param_path = None if calib_param_path is None else str(calib_param_path)

    _validate_inputs()

    y_pred = np.full(len(y), np.nan, dtype=np.float32)
    y_pred_ac = np.full(len(y), np.nan, dtype=np.float32) if apply_correction else None
    fold_n = np.full(len(y), -1, dtype=np.int8)
    perfs = []
    best_score = np.inf if perf_metrix == "MAE" else -np.inf

    if n_folds > 1:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        pipes, calibs = {}, {}
        for k, (tr, va) in enumerate(kf.split(idx_tr)):
            pipes[k] = _fit_or_load(
                suffix=f"fold-{k}",
                model=_init_model(),
                X_fit=_get_rows(idx_tr[tr]), 
                y_fit=y[idx_tr[tr]]
            )
            y_pred[idx_tr[va]] = pipes[k].predict(_get_rows(idx_tr[va]))

            if apply_correction:
                calibs[k] = _fit_calibrator(y[idx_tr[va]], y_pred[idx_tr[va]])
                y_pred_ac[idx_tr[va]] = (
                    (y_pred[idx_tr[va]] - calibs[k]["intercept"]) / calibs[k]["slope"] 
                )

            fold_n[idx_tr[va]] = k

            perf_va = _calc_model_perf(y[idx_tr[va]], y_pred[idx_tr[va]])
            score = perf_va[perf_metrix]

            improved = (score < best_score) if perf_metrix == "MAE" else (score > best_score)
            if improved:
                best_score = score
                best_fold = k

            if model_perf_path:
                perfs.append(
                    {"Split": f"Val_fold-{k}", "N": len(idx_tr[va]), "MAE": perf_va["MAE"], "R2": perf_va["R2"]}
                )

        best_pipe = pipes[best_fold]
        y_pred[idx_te] = best_pipe.predict(_get_rows(idx_te))

        if apply_correction:
            best_calib = calibs[best_fold]
            y_pred_ac[idx_te] = (
                (y_pred[idx_te] - best_calib["intercept"]) / best_calib["slope"]
            )

        if model_perf_path:
            fold_mae = np.array([ p["MAE"] for p in perfs ])
            fold_r2  = np.array([ p["R2"] for p in perfs ])
            perf_tr = _calc_model_perf(y[idx_tr], y_pred[idx_tr])
            perf_te = _calc_model_perf(y[idx_te], y_pred[idx_te])
            perfs += [
                {"Split": "Val_mean"  , "N": None       , "MAE": fold_mae.mean()     , "R2": fold_r2.mean()},
                {"Split": "Val_SD"    , "N": None       , "MAE": fold_mae.std(ddof=1), "R2": fold_r2.std(ddof=1)},
                {"Split": "Val_pooled", "N": len(idx_tr), "MAE": perf_tr["MAE"]      , "R2": perf_tr["R2"]}, 
                {"Split": "Test"      , "N": len(idx_te), "MAE": perf_te["MAE"]      , "R2": perf_te["R2"]}
            ]

    else: # both predict by the model train on entire training set
        pipe = _fit_or_load(
            suffix="all",
            model=_init_model(),
            X_fit=_get_rows(idx_tr),
            y_fit=y[idx_tr]
        )
        y_pred[idx_tr] = pipe.predict(_get_rows(idx_tr))
        fold_n[idx_tr] = 0

        y_pred[idx_te] = pipe.predict(_get_rows(idx_te))

        if model_perf_path:
            perf_tr = _calc_model_perf(y[idx_tr], y_pred[idx_tr])
            perf_te = _calc_model_perf(y[idx_te], y_pred[idx_te])
            perfs += [
                {"Split": "Train", "N": len(idx_tr), "MAE": perf_tr["MAE"], "R2": perf_tr["R2"]},
                {"Split": "Test" , "N": len(idx_te), "MAE": perf_te["MAE"], "R2": perf_te["R2"]}
            ]
        
    if model_perf_path:
        perfs = pd.DataFrame(perfs)
        perfs["N"] = perfs["N"].astype("Int64")  # nullable int
        perfs.to_csv(model_perf_path, index=False)
        custom_print(f"\nSaved: {model_perf_path}")
        custom_print(f"Test MAE = {perf_te["MAE"]:.1f}")

    if calib_param_path:
        calib_df = pd.DataFrame.from_dict(calibs, orient="index")
        calib_df.index.name = "Fold"
        calib_df.to_csv(calib_param_path)
        custom_print(f"\nSaved: {calib_param_path}")

    return y_pred, y_pred_ac, fold_n