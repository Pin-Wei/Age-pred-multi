#!/usr/bin/env python3


import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import custom_print


MODEL_TYPES = ["elasticnet", "lasso", "ridge", "xgboost"]
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
ALPHAS = [1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5]
XGB_PARAM_SPACE = {
    "max_depth"       : lambda t: t.suggest_int("max_depth", 1, 9),
    "learning_rate"   : lambda t: t.suggest_float("learning_rate", 1e-4, 1.0, log=True),
    "n_estimators"    : lambda t: t.suggest_int("n_estimators", 100, 1000),
    "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
    "subsample"       : lambda t: t.suggest_float("subsample", 0.1, 1.0),
    "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.1, 1.0),
    # "max_bin"         : lambda t: t.suggest_int("max_bin", 32, 128),
}


def calc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
) -> dict[str, float]:
    '''
    Calculate performance metrics (MAE, R^2) for each target.
    '''
    perf: dict[str, float] = {}

    for i, target in enumerate(targets):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        err = yt - yp
        mae = float(np.mean(np.abs(err)))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan
        perf[f"MAE_{target}"] = mae
        perf[f"R2_{target}"] = r2

    return perf


def fit_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
) -> dict[str, float]:
    '''
    Fit simple linear regression of predictions on true values per target
    (to perform Cole's age-bias correction):
    y_pred = const + y_true * slope
    '''
    coefs_out: dict[str, float] = {}

    for i, target in enumerate(targets):
        slope, intercept = np.polyfit(y_true[:, i], y_pred[:, i], 1)
        coefs_out[f"const_{target}"] = float(intercept)
        coefs_out[f"slope_{target}"] = float(slope)

    return coefs_out


def apply_calibrator(
    y_pred: np.ndarray,
    coefs: dict[str, float],
    targets: list[str],
) -> np.ndarray:
    '''
    Apply Cole's age-bias correction by removing the baseline shift 
    and reversing the regression-to-the-mean compression:
    corrected_pred = (y_pred - const) / slope
    '''
    corrected = np.empty_like(y_pred)

    for i, target in enumerate(targets):
        slope = coefs[f"slope_{target}"]
        if np.isclose(slope, 0.0):
            corrected[:, i] = np.nan
        else:
            corrected[:, i] = (y_pred[:, i] - coefs[f"const_{target}"]) / slope

    return corrected


def xgb_cuda_available() -> bool:
    try:
        import json
        import warnings
        import xgboost as xgb

        if not xgb.build_info().get("USE_CUDA", False):
            return False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
            reg.fit([[0.0]], [0.0])

        cfg = json.loads(reg.get_booster().save_config())
        device = cfg.get("learner", {}).get("generic_param", {}).get("device", "")

        return device.startswith("cuda")

    except Exception:
        return False


def validate_inputs(
    X: np.ndarray | pd.DataFrame,
    Y: pd.DataFrame,
    idx_tr: np.ndarray,
    idx_te: np.ndarray,
    model_type: str,
    perf_metric: str,
    xgb_device: str,
    n_folds: int,
    apply_correction: bool,
    model_path_template: str | None,
    model_perf_path: str | None,
    calib_param_path: str | None,
) -> None:

    if X.shape[0] != len(Y):
        raise ValueError(f"Mismatch between length of X ({X.shape[0]}) and y ({len(Y)}).")
    if Y.shape[1] <= 0:
        raise ValueError("y carries no target.")
    if not Y.dtypes.map(pd.api.types.is_numeric_dtype).all():
        raise ValueError("y carries non-numeric target(s).")
    if Y.isna().any(axis=None):
        raise ValueError("y carries NaN target(s).")
    if idx_tr.dtype.kind not in "iu" or idx_te.dtype.kind not in "iu":
        raise ValueError(f"idx_tr/idx_te must be integer indices, got {idx_tr.dtype}/{idx_te.dtype}.")

    overlap = np.intersect1d(idx_tr, idx_te)
    if overlap.size != 0:
        raise ValueError(f"{overlap.size} overlap(s) between training and testing indices.")

    for nm, idx in [("idx_tr", idx_tr), ("idx_te", idx_te)]:
        if idx.size == 0:
            raise ValueError(f"{nm} is empty.")
        if idx.min() < 0 or idx.max() >= len(Y):
            raise ValueError(f"{nm} out of bounds: [{idx.min()}, {idx.max()}] vs len(y)={len(Y)}.")

    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model type '{model_type}' is undefined.")
    if perf_metric not in {"MAE", "R2"}:
        raise ValueError(f"Performance metric '{perf_metric}' is undefined.")
    if xgb_device not in {"cpu", "cuda"}:
        raise ValueError(f"Device '{xgb_device}' is invalid.")
    if apply_correction and n_folds <= 1:
        raise ValueError(
            "'apply_correction' fits one slope per fold on that fold's held-out rows, "
            f"but n_folds is {n_folds}; there are no held-out rows to fit on."
        )

    if model_path_template:
        if "{}" not in model_path_template:
            raise ValueError(f"'model_path_template' should have '{{}}', got: {model_path_template}")
        if not model_path_template.endswith(".joblib"):
            raise ValueError(f"'model_path_template' should end with .joblib, got: {model_path_template}")
        parent = os.path.dirname(model_path_template.format("x"))
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        custom_print(
            "\n'model_path_template' is not defined. No trained model will be saved.\n", 
            level="WARNING"
        )

    if model_perf_path:
        if not model_perf_path.endswith(".csv"):
            raise ValueError(f"'model_perf_path' should end with .csv, got: {model_perf_path}")
        parent = os.path.dirname(model_perf_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        custom_print(
            "\n'model_perf_path' is not defined. Model performances will not be calculated and saved.\n", 
            level="WARNING"
        )

    if calib_param_path:
        if not apply_correction:
            raise ValueError("'calib_param_path' is given but 'apply_correction' is False.")
        if not calib_param_path.endswith(".csv"):
            raise ValueError(f"'calib_param_path' should end with .csv, got: {calib_param_path}")
        parent = os.path.dirname(calib_param_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def build_pipeline(
    model_type: str,
    n_features: int,
    n_targets: int,
    *,
    seed_inner: int = 0,
    l1_ratios: list[float] | None = None,
    alphas: list[float] | None = None,
    max_iter: int = 10000,
    xgb_params: dict | None = None,
    device: str = "cpu",
    n_jobs: int = -1,
    verbose: int = 1,
    impute_data: bool = False,
):
    '''
    Construct the [optional imputer -> optional scaler -> regressor] sklearn Pipeline.
    '''
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import (
        ElasticNetCV,
        LassoCV,
        MultiTaskElasticNetCV,
        MultiTaskLassoCV,
        RidgeCV,
    )
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    multi = n_targets > 1
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=seed_inner)
    l1_ratios = list(L1_RATIOS) if l1_ratios is None else list(l1_ratios)
    alphas = list(ALPHAS) if alphas is None else list(alphas)
    xgb_params = {} if xgb_params is None else dict(xgb_params)

    if model_type == "elasticnet":
        regressor = (MultiTaskElasticNetCV if multi else ElasticNetCV)(
            l1_ratio=l1_ratios,
            alphas=alphas,
            cv=kf_inner,
            max_iter=max_iter,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    elif model_type == "lasso":
        regressor = (MultiTaskLassoCV if multi else LassoCV)(
            alphas=alphas,
            cv=kf_inner,
            max_iter=max_iter,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    elif model_type == "ridge":
        regressor = RidgeCV(
            alphas=alphas,
            cv=kf_inner,
        )
    elif model_type == "xgboost":
        from xgboost import XGBRegressor
        regressor = XGBRegressor(
            **xgb_params,
            multi_strategy="multi_output_tree" if multi else "one_output_per_tree",
            random_state=seed_inner,
            device=device,
            n_jobs=n_jobs,
            verbosity=verbose,
        )
    else:
        raise ValueError(f"Model type '{model_type}' is undefined.")

    if multi:
        from models import TargetScaler
        regressor = TransformedTargetRegressor(
            regressor=regressor, transformer=TargetScaler()
        )

    steps = []
    if impute_data:
        steps.append(("imputer", SimpleImputer(strategy="median")))

    if model_type != "xgboost":  # tree models are scale-invariant; avoids copying huge float64 matrices
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", regressor))

    return Pipeline(steps=steps)


def optimize_xgb_params(
    X_tr: np.ndarray | pd.DataFrame,
    y_tr: np.ndarray,
    n_features: int,
    n_targets: int,
    *,
    xgb_params: dict,
    opt_trials: int,
    seed_inner: int,
    n_jobs: int,
    xgb_device: str,
    verbose: int,
    impute_data: bool,
) -> dict:
    import optuna
    from sklearn.metrics import make_scorer, mean_absolute_error
    from sklearn.model_selection import KFold, cross_val_score

    if n_targets > 1:
        stds = np.nanstd(y_tr, axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        target_scales = stds / stds[0]
        scoring = make_scorer(
            mean_absolute_error,
            greater_is_better=False,
            multioutput=1.0 / target_scales,
        )
    else:
        scoring = "neg_mean_absolute_error"

    def _eval_trial(trial, device: str) -> float:
        sampled = { k: fun(trial) for k, fun in XGB_PARAM_SPACE.items() if k not in xgb_params }
        merged_params = {**xgb_params, **sampled}
        outer_jobs = 1 if device == "cuda" else n_jobs
        inner_jobs = n_jobs if outer_jobs == 1 else 1  # avoid nested parallel processing

        try:
            neg_mae_scores = cross_val_score(
                estimator=build_pipeline(
                    "xgboost",
                    n_features,
                    n_targets,
                    seed_inner=seed_inner,
                    xgb_params=merged_params,
                    device=device,
                    n_jobs=inner_jobs,
                    verbose=verbose,
                    impute_data=impute_data,
                ),
                X=X_tr,
                y=y_tr,
                cv=KFold(n_splits=5, shuffle=True, random_state=seed_inner),
                scoring=scoring,
                n_jobs=outer_jobs,
                error_score="raise" if device == "cuda" else np.nan,
                verbose=verbose,
            )
        except Exception as e:
            if (device == "cuda") and ("out of memory" in str(e).lower()):
                custom_print(
                    f"\nTrial {trial.number} ran out of GPU memory; evaluating it on the CPU instead.",
                    level="WARNING",
                )
                return _eval_trial(trial, device="cpu")
            raise

        return float(-1.0 * np.mean(neg_mae_scores))

    # try:
    #     module = optunahub.load_module(package="samplers/auto_sampler")
    #     sampler = module.AutoSampler(seed=seed_inner)
    # except Exception as e:
    #     reason = str(e).splitlines()[0] if str(e) else type(e).__name__
    #     custom_print(
    #         f"\nFalling back on TPESampler, AutoSampler is unavailable: {reason}", 
    #         level="WARNING"
    #     )
    #     sampler = optuna.samplers.TPESampler(seed=seed_inner)
    sampler = optuna.samplers.TPESampler(seed=seed_inner)

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose > 0 else optuna.logging.WARNING)
    custom_print(f"\nSearching the XGBoost hyperparameters over {opt_trials} trial(s) ...")
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: _eval_trial(trial, xgb_device), n_trials=opt_trials)

    custom_print("\nHyperparameter optimization is completed :-)")
    custom_print(f"Cross-validated MAE of the best trial: {study.best_value:.3f}")
    custom_print("\nBest XGBoost hyperparameters = {")
    for k, v in study.best_params.items():
        custom_print(f"\t{k}: {v}")
    custom_print("}")

    return study.best_params


def resolve_xgb_params(
    X_tr: np.ndarray | pd.DataFrame,
    y_tr: np.ndarray,
    n_features: int,
    n_targets: int,
    *,
    xgb_params: dict,
    opt_trials: int,
    n_folds: int,
    seed_inner: int,
    n_jobs: int,
    xgb_device: str,
    verbose: int,
    impute_data: bool,
    model_path_template: str | None,
    overwrite: bool,
) -> tuple[dict, str]:
    '''
    Verify CUDA availability and resolve tunable XGBoost hyperparameters once on
    the training split (either fixed in `xgb_params`, loaded from a cached fold
    model, or searched via Optuna). Returns `(best_xgb_params, resolved_device)`.
    '''
    import joblib

    if xgb_device == "cuda" and not xgb_cuda_available():
        custom_print(
            "\nCUDA is unavailable for XGBoost; falling back to CPU.", 
            level="WARNING"
        )
        xgb_device = "cpu"

    if xgb_params:
        custom_print("\nFixed XGBoost hyperparameters = {")
        for k, v in xgb_params.items():
            custom_print(f"\t{k}: {v},")
        custom_print("}")

    if not (XGB_PARAM_SPACE.keys() - xgb_params.keys()):
        custom_print("\n'xgb_params' fixes every tunable hyperparameter; no search is needed.\n")
        return {}, xgb_device

    if (not overwrite) and bool(model_path_template):
        suffixes = [ f"fold-{k}" for k in range(n_folds) ] if n_folds > 1 else ["all"]
        for s in suffixes:
            model_path = model_path_template.format(s)
            if os.path.isfile(model_path):
                custom_print("\nTrained model exists; the hyperparameter search is skipped.\n")
                model = joblib.load(model_path).named_steps["model"]
                model = getattr(model, "regressor_", model)
                params = model.get_params()
                cached = { k: params[k] for k in XGB_PARAM_SPACE if k not in xgb_params }
                return cached, xgb_device

    if opt_trials <= 0:
        raise ValueError("To perform hyperparameter tuning, `opt_trials` must be greater than zero.")

    best_params = optimize_xgb_params(
        X_tr,
        y_tr,
        n_features,
        n_targets,
        xgb_params=xgb_params,
        opt_trials=opt_trials,
        seed_inner=seed_inner,
        n_jobs=n_jobs,
        xgb_device=xgb_device,
        verbose=verbose,
        impute_data=impute_data,
    )

    return best_params, xgb_device


def train_eval_model(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame,
    idx_tr: np.ndarray,
    idx_te: np.ndarray,
    model_type: str,
    seed: int = 42,
    seed_inner: int = 0,
    n_folds: int = 5,
    l1_ratios: list[float] | None = None,
    alphas: list[float] | None = None,
    max_iter: int = 10000,
    xgb_params: dict | None = None,
    opt_trials: int = 100,
    n_jobs: int = -1,
    xgb_device: str = "cpu",
    verbose: int = 1,
    impute_data: bool = False,
    perf_metric: str = "MAE",
    apply_correction: bool = False,
    model_path_template: str | Path | None = None,
    model_perf_path: str | Path | None = None,
    calib_param_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray]:
    '''
    Fit one or more [optional imputer -> optional scaler -> regressor] pipeline(s)
    to produce train/test predictions, their age-corrected counterparts (optional),
    and outer fold indices.
    '''
    import joblib
    from sklearn.model_selection import KFold

    l1_ratios = list(L1_RATIOS) if l1_ratios is None else list(l1_ratios)
    alphas = list(ALPHAS) if alphas is None else list(alphas)
    xgb_params = {} if xgb_params is None else dict(xgb_params)

    model_path_template = None if model_path_template is None else str(model_path_template)
    model_perf_path = None if model_perf_path is None else str(model_perf_path)
    calib_param_path = None if calib_param_path is None else str(calib_param_path)

    Y = y if isinstance(y, pd.DataFrame) else pd.DataFrame(np.asarray(y).reshape(len(y), -1)).add_prefix("y")
    Y = Y.reset_index(drop=True)  # indices below are positional
    targets = Y.columns.to_list()
    n_targets = len(targets)
    y_arr = Y.to_numpy(dtype=np.float64)
    n_samples = len(y_arr)
    n_features = X.shape[1]

    idx_tr = np.asarray(idx_tr)
    idx_te = np.asarray(idx_te)

    validate_inputs(
        X, Y, idx_tr, idx_te,
        model_type=model_type,
        perf_metric=perf_metric,
        xgb_device=xgb_device,
        n_folds=n_folds,
        apply_correction=apply_correction,
        model_path_template=model_path_template,
        model_perf_path=model_perf_path,
        calib_param_path=calib_param_path,
    )

    def _get_features(idx: np.ndarray) -> np.ndarray | pd.DataFrame:
        return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]

    def _get_targets(idx: np.ndarray) -> np.ndarray:
        return y_arr[idx] if n_targets > 1 else y_arr[idx, 0]

    def _make_pipe(device: str | None = None):
        # Resolve `xgb_device` at call time so CUDA->CPU fallback takes effect
        resolved_device = xgb_device if device is None else device

        return build_pipeline(
            model_type=model_type,
            n_features=n_features,
            n_targets=n_targets,
            seed_inner=seed_inner,
            l1_ratios=l1_ratios,
            alphas=alphas,
            max_iter=max_iter,
            xgb_params={**xgb_params, **best_xgb_params},
            device=resolved_device,
            n_jobs=n_jobs,
            verbose=verbose,
            impute_data=impute_data,
        )

    def _fit_or_load(suffix: str, idx: np.ndarray):
        model_path = model_path_template.format(suffix) if bool(model_path_template) else None
        model_is_cached = bool(model_path) and os.path.isfile(model_path) and not overwrite

        if model_is_cached:
            pipeline = joblib.load(model_path)
            cached_targets = list(getattr(pipeline, "target_names_in_", []))
            if cached_targets != targets:
                raise ValueError(
                    f"{os.path.basename(model_path)} was fit on {cached_targets or 'unrecorded target(s)'}, "
                    f"not on {targets}. Refit it by setting 'overwrite' to True."
                )
            custom_print(f"Loaded pre-trained: {model_path}")
            return pipeline

        pipeline = _make_pipe()
        X_fit, y_fit = _get_features(idx), _get_targets(idx)

        try:
            pipeline.fit(X_fit, y_fit)
        except Exception as e:
            if (model_type == "xgboost") and (xgb_device == "cuda") and ("out of memory" in str(e).lower()):
                custom_print(
                    f"\nThe '{suffix}' model ran out of GPU memory; fitting it on the CPU instead.", 
                    level="WARNING"
                )
                pipeline = _make_pipe(device="cpu")
                pipeline.fit(X_fit, y_fit)
            else:
                raise

        pipeline.target_names_in_ = targets  # self-define property

        if model_path:
            joblib.dump(pipeline, model_path)
            custom_print(f"\nSaved: {model_path}\n")

        return pipeline

    def _predict(pipeline, idx: np.ndarray) -> np.ndarray:
        return np.asarray(
            pipeline.predict(_get_features(idx))
        ).reshape(len(idx), n_targets)

    ## Initialization -----------------------------------------------------
    y_pred = np.full((n_samples, n_targets), np.nan, dtype=np.float32)
    y_pred_ac = np.full_like(y_pred, np.nan) if apply_correction else None
    fold_n = np.full(n_samples, -1, dtype=np.int32)
    perfs = []
    best_score = np.inf if perf_metric == "MAE" else -np.inf

    best_xgb_params, xgb_device = (
        resolve_xgb_params(
            _get_features(idx_tr),
            _get_targets(idx_tr),
            n_features,
            n_targets,
            xgb_params=xgb_params,
            opt_trials=opt_trials,
            n_folds=n_folds,
            seed_inner=seed_inner,
            n_jobs=n_jobs,
            xgb_device=xgb_device,
            verbose=verbose,
            impute_data=impute_data,
            model_path_template=model_path_template,
            overwrite=overwrite,
        )
        if model_type == "xgboost"
        else ({}, xgb_device)
    )

    if n_folds > 1:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        best_pipe = None
        best_fold = 0
        calibs: dict[int, dict[str, float]] = {}

        for k, (tr, va) in enumerate(kf.split(idx_tr)):
            pipe_k = _fit_or_load(f"fold-{k}", idx_tr[tr])
            y_pred[idx_tr[va]] = _predict(pipe_k, idx_tr[va])
            fold_n[idx_tr[va]] = k

            if apply_correction:
                calibs[k] = fit_calibrator(y_arr[idx_tr[va]], y_pred[idx_tr[va]], targets)
                y_pred_ac[idx_tr[va]] = apply_calibrator(y_pred[idx_tr[va]], calibs[k], targets)

            perf_va = calc_metrics(y_arr[idx_tr[va]], y_pred[idx_tr[va]], targets)
            score = float(np.nanmean([ perf_va[f"{perf_metric}_{t}"] for t in targets ]))

            improved = (
                best_pipe is None
                or (not np.isnan(score) and ((score < best_score) if perf_metric == "MAE" else (score > best_score)))
            )
            if improved:
                best_score = score
                best_fold = k
                best_pipe = pipe_k  # retain only the winning pipeline to save memory

            if model_perf_path:
                perfs.append({"Split": f"Val_fold-{k}", "N": len(idx_tr[va]), **perf_va})

        y_pred[idx_te] = _predict(best_pipe, idx_te)

        if apply_correction:
            y_pred_ac[idx_te] = apply_calibrator(y_pred[idx_te], calibs[best_fold], targets)

        if model_perf_path:
            fold_perfs = pd.DataFrame(perfs).drop(columns=["Split", "N"])
            perf_tr = calc_metrics(y_arr[idx_tr], y_pred[idx_tr], targets)
            perf_te = calc_metrics(y_arr[idx_te], y_pred[idx_te], targets)
            perfs += [
                {"Split": "Val_mean",   "N": None,        **fold_perfs.mean().to_dict()},
                {"Split": "Val_SD",     "N": None,        **fold_perfs.std(ddof=1).to_dict()},
                {"Split": "Val_pooled", "N": len(idx_tr), **perf_tr},
                {"Split": "Test",       "N": len(idx_te), **perf_te},
            ]

    else:  # both predict by the model trained on entire training set
        pipe = _fit_or_load("all", idx_tr)
        y_pred[idx_tr] = _predict(pipe, idx_tr)
        fold_n[idx_tr] = 0
        y_pred[idx_te] = _predict(pipe, idx_te)

        if model_perf_path:
            perf_tr = calc_metrics(y_arr[idx_tr], y_pred[idx_tr], targets)
            perf_te = calc_metrics(y_arr[idx_te], y_pred[idx_te], targets)
            perfs += [
                {"Split": "Train", "N": len(idx_tr), **perf_tr},
                {"Split": "Test",  "N": len(idx_te), **perf_te},
            ]

    if model_perf_path:
        perfs_df = pd.DataFrame(perfs)
        perfs_df["N"] = perfs_df["N"].astype("Int64")  # nullable int
        perfs_df.to_csv(model_perf_path, index=False)
        custom_print(f"\nSaved: {model_perf_path}")
        custom_print("Test MAE: " + ", ".join(f"{t} = {perf_te[f'MAE_{t}']:.2f}" for t in targets))

    if calib_param_path:
        calib_df = pd.DataFrame.from_dict(calibs, orient="index")
        calib_df.index.name = "Fold"
        calib_df.to_csv(calib_param_path)
        custom_print(f"\nSaved: {calib_param_path}")

    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    y_pred_ac_df = pd.DataFrame(y_pred_ac, columns=targets) if apply_correction else None

    return y_pred_df, y_pred_ac_df, fold_n