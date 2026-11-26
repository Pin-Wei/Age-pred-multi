#!/usr/bin/env python3


import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import custom_print


MODEL_TYPES = ["elasticnet", "lasso", "ridge", "xgboost", "msvr", "mogp", "pls"]
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
ALPHAS = [1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5]
# ALPHAS_LOWER = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 3.0, 1e1, 1e2, 1e3]
# ALPHAS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5]
XGB_PARAM_SPACE = {
    "max_depth"       : lambda t: t.suggest_int("max_depth", 1, 9),
    "learning_rate"   : lambda t: t.suggest_float("learning_rate", 1e-4, 1.0, log=True),
    "n_estimators"    : lambda t: t.suggest_int("n_estimators", 100, 1000),
    "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
    "subsample"       : lambda t: t.suggest_float("subsample", 0.1, 1.0),
    "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.1, 1.0),
    # "max_bin"         : lambda t: t.suggest_int("max_bin", 32, 128),
}
MSVR_GRID = {
    "C"      : [1e-1, 1e0, 1e1, 1e2],
    "epsilon": [0.1, 0.5],
    "gamma"  : [0.1, 1.0, 10.0],
}
DEFAULT_PLS_MAX_K = 20


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


def extract_pls_estimator(pipeline):
    '''
    Unwrap Pipeline -> [TransformedTargetRegressor] -> [GridSearchCV] -> PLSRegression.
    '''
    est = pipeline.named_steps["model"]
    est = getattr(est, "regressor_", est)
    est = getattr(est, "best_estimator_", est)

    return est


def transform_pls_components(pipeline, X_sub) -> np.ndarray:
    '''
    Apply all preprocessing steps before `model`,
    then project onto the fitted PLS latent components.
    '''
    X_t = X_sub
    for _, step in pipeline.steps[:-1]:
        X_t = step.transform(X_t)

    pls = extract_pls_estimator(pipeline)

    return np.asarray(pls.transform(X_t), dtype=np.float32)


def select_pls_components_pa(
    X_tr: np.ndarray | pd.DataFrame,
    y_tr: np.ndarray,
    max_components: int = DEFAULT_PLS_MAX_K,
    n_perm: int = 100,
    percentile: float = 95.0,
    cv_splits: int = 5,
    random_state: int = 0,
    impute_data: bool = False,
) -> int:
    '''
    Select the number of supervised PLS components 
    via cross-validated permutation Parallel Analysis.

    Using out-of-fold incremental R^2 (instead of in-sample Y-variance) avoids
    high-dimensional (p >> n) overfitting inflation in the permuted null distribution.
    A thin SVD row-space projection is applied when `p > n` so every CV fit runs
    in O(n^2 * k) with mathematically identical PLS scores and predictions.
    '''
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    X_arr = np.asarray(X_tr, dtype=np.float64)
    if impute_data:
        X_arr = SimpleImputer(strategy="median").fit_transform(X_arr)
    X_arr = StandardScaler().fit_transform(X_arr)

    Y_arr = np.asarray(y_tr, dtype=np.float64).reshape(len(X_arr), -1)
    Y_arr = StandardScaler().fit_transform(Y_arr)

    n_samples, n_features = X_arr.shape
    fold_tr_min = n_samples - int(np.ceil(n_samples / cv_splits))
    k_max = int(min(max_components, fold_tr_min - 1, n_features))
    if k_max <= 1:
        return 1

    # Row-space reduction when p > n: Z @ Z.T == X @ X.T, preserving exact NIPALS PLS scores
    if n_features > n_samples:
        U, s, _ = np.linalg.svd(X_arr, full_matrices=False)
        Z = U * s
    else:
        Z = X_arr

    folds = list(KFold(n_splits=cv_splits, shuffle=True, random_state=random_state).split(Z))

    def _oof_delta_r2(Y_mat: np.ndarray) -> np.ndarray:
        oof_preds = np.empty((k_max, n_samples, Y_mat.shape[1]), dtype=np.float64)
        for tr, va in folds:
            Z_tr, Z_va = Z[tr], Z[va]
            Y_fold_tr = Y_mat[tr]
            pls = PLSRegression(n_components=k_max, scale=False)
            pls.fit(Z_tr, Y_fold_tr)

            T_va = (Z_va - pls._x_mean) @ pls.x_rotations_  # (n_va, k_max)
            Q = pls.y_loadings_                              # (q, k_max)
            cum_pred = np.broadcast_to(pls._y_mean, (len(va), Y_mat.shape[1])).copy()
            for k in range(k_max):
                cum_pred = cum_pred + np.outer(T_va[:, k], Q[:, k])
                oof_preds[k, va] = cum_pred

        total_ss = float(np.sum((Y_mat - Y_mat.mean(axis=0)) ** 2))
        if total_ss <= 0:
            return np.zeros(k_max, dtype=np.float64)

        cum_r2 = np.array([
            1.0 - float(np.sum((Y_mat - oof_preds[k]) ** 2)) / total_ss
            for k in range(k_max)
        ])
        prev_r2 = np.concatenate(([0.0], cum_r2[:-1]))

        return cum_r2 - prev_r2

    obs_delta = _oof_delta_r2(Y_arr)
    rng = np.random.default_rng(random_state)
    null_delta = np.empty((n_perm, k_max), dtype=np.float64)
    for b in range(n_perm):
        perm_idx = rng.permutation(n_samples)
        null_delta[b] = _oof_delta_r2(Y_arr[perm_idx])

    thresh = np.percentile(null_delta, percentile, axis=0)
    above = (obs_delta > thresh) & (obs_delta > 0.0)

    # Keep the longest leading prefix above both 0 and the permutation threshold
    n_selected = 0
    for is_sig in above:
        if is_sig:
            n_selected += 1
        else:
            break

    if n_selected == 0:
        custom_print(
            f"OOF parallel analysis found no PLS component above the {percentile:.0f}th permutation percentile; "
            "falling back to 1 component.",
            level="WARNING",
        )
        n_selected = 1
    else:
        custom_print(
            f"OOF parallel analysis ({n_perm} permutations, {percentile:.0f}th pct) selected "
            f"{n_selected} / {k_max} PLS component(s) "
            f"(OOF R^2 = {obs_delta[:n_selected].sum():.3f}; "
            f"last dR^2 = {obs_delta[n_selected - 1]:.4f} vs thr {thresh[n_selected - 1]:.4f})."
        )

    return n_selected


def resolve_pls_components(
    X_tr: np.ndarray | pd.DataFrame,
    y_tr_2d: np.ndarray,
    *,
    pls_components: int | str | None,
    pls_max_components: int,
    pls_n_perm: int,
    return_components: bool,
    n_folds: int,
    seed_inner: int,
    n_jobs: int,
    verbose: int,
    impute_data: bool,
    model_path_template: str | None,
    overwrite: bool,
) -> int | None:
    '''
    Determine the PLS component count once on the training split when fixed,
    selected by parallel analysis ("auto"), or required to be uniform across
    outer folds (`return_components=True`). Returns None when each fold should
    tune its own component count via inner CV.
    '''
    import joblib

    n_train, n_features = X_tr.shape
    n_targets = y_tr_2d.shape[1]
    fold_tr_min = (n_train - int(np.ceil(n_train / n_folds))) if n_folds > 1 else n_train
    max_valid_k = max(1, min(fold_tr_min - 1, n_features))

    def _find_cached_k() -> int | None:
        if overwrite or not model_path_template:
            return None
        suffixes = [ f"fold-{k}" for k in range(n_folds) ] if n_folds > 1 else ["all"]
        for s in suffixes:
            mp = model_path_template.format(s)
            if os.path.isfile(mp):
                return int(extract_pls_estimator(joblib.load(mp)).n_components)
        return None

    if isinstance(pls_components, int):
        if pls_components < 1:
            raise ValueError(f"`pls_components` must be >= 1, got {pls_components}.")
        return min(pls_components, max_valid_k)

    if pls_components == "auto":
        cached_k = _find_cached_k()
        if cached_k is not None:
            custom_print(f"\nLoaded PLS component count ({cached_k}) from cached model.")
            return cached_k

        return select_pls_components_pa(
            X_tr,
            y_tr_2d,
            max_components=min(pls_max_components, max_valid_k),
            n_perm=pls_n_perm,
            random_state=seed_inner,
            impute_data=impute_data,
        )

    if pls_components in {None, "cv"}:
        if not return_components:
            return None

        cached_k = _find_cached_k()
        if cached_k is not None:
            custom_print(f"\nLoaded PLS component count ({cached_k}) from cached model.")
            return cached_k

        probe = build_pipeline(
            "pls",
            n_features,
            n_targets,
            seed_inner=seed_inner,
            pls_n_components=None,
            pls_max_components=min(pls_max_components, max_valid_k),
            n_jobs=n_jobs,
            verbose=verbose,
            impute_data=impute_data,
        )
        y_fit = y_tr_2d if n_targets > 1 else y_tr_2d[:, 0]
        probe.fit(X_tr, y_fit)
        best_k = int(extract_pls_estimator(probe).n_components)
        custom_print(f"\nInner CV selected {best_k} PLS component(s) for cross-decomp.")

        return best_k

    raise ValueError(
        f"`pls_components` must be an int, 'auto', 'cv', or None; got {pls_components!r}."
    )


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
    return_components: bool,
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
    if return_components and model_type != "pls":
        raise ValueError(f"`return_components=True` is only supported for model_type='pls', got '{model_type}'.")
    if return_components and n_folds <= 1:
        raise ValueError(
            "`return_components=True` requires n_folds > 1 to produce out-of-fold components "
            f"without target leakage, got n_folds={n_folds}."
        )
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
    pls_n_components: int | None = None,
    pls_max_components: int = DEFAULT_PLS_MAX_K,
    device: str = "cpu",
    n_jobs: int = -1,
    verbose: int = 1,
    impute_data: bool = False,
):
    '''
    Construct the [optional imputer -> optional scaler -> regressor] sklearn Pipeline.
    Custom estimators from `models.py` are imported lazily
    to avoid loading PyTorch/GPyTorch unless actually needed.
    '''
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import (
        ElasticNetCV,
        LassoCV,
        MultiTaskElasticNetCV,
        MultiTaskLassoCV,
        RidgeCV,
    )
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    multi = n_targets > 1
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=seed_inner)
    l1_ratios = list(L1_RATIOS) if l1_ratios is None else list(l1_ratios)
    alphas = list(ALPHAS) if alphas is None else list(alphas)
    xgb_params = {} if xgb_params is None else dict(xgb_params)

    def _grid_search(estimator, grid: dict):
        return GridSearchCV(
            estimator,
            grid,
            scoring="neg_mean_absolute_error",
            cv=kf_inner,
            n_jobs=n_jobs,
            verbose=verbose,
        )

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
    elif model_type == "msvr":
        from models import MSVR
        regressor = _grid_search(
            MSVR(),
            {**MSVR_GRID, "gamma": [ g / n_features for g in MSVR_GRID["gamma"] ]},
        )
    elif model_type == "mogp":
        from models import MOGP
        regressor = MOGP(
            random_state=seed_inner,
        )
    elif model_type == "pls":
        if pls_n_components is not None:
            k_fixed = int(min(pls_n_components, n_features))
            regressor = PLSRegression(n_components=max(1, k_fixed))
        else:
            max_k = min(pls_max_components, n_features)
            regressor = _grid_search(
                PLSRegression(),
                {"n_components": list(range(1, max(2, max_k + 1)))},
            )
    else:
        raise ValueError(f"Model type '{model_type}' is undefined.")

    if model_type in {"msvr", "mogp"}:  # sensitive to target scale
        regressor = TransformedTargetRegressor(
            regressor=regressor, transformer=StandardScaler()
        )
    elif multi:
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
    pls_components: int | str | None = None,
    pls_max_components: int = DEFAULT_PLS_MAX_K,
    pls_n_perm: int = 100,
    return_components: bool = False,
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
) -> (
    tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray]
    | tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray, pd.DataFrame]
):
    '''
    Fit one or more [optional imputer -> optional scaler -> regressor] pipeline(s)
    to produce train/test predictions, their age-corrected counterparts (optional),
    outer fold indices, and (when `return_components=True` for PLS) sign-aligned
    out-of-fold / test latent components (`PLS1`, ..., `PLSK`).
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
        return_components=return_components,
        model_path_template=model_path_template,
        model_perf_path=model_perf_path,
        calib_param_path=calib_param_path,
    )

    def _get_features(idx: np.ndarray) -> np.ndarray | pd.DataFrame:
        return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]

    def _get_targets(idx: np.ndarray) -> np.ndarray:
        return y_arr[idx] if n_targets > 1 else y_arr[idx, 0]

    resolved_pls_k = (
        resolve_pls_components(
            _get_features(idx_tr),
            y_arr[idx_tr],
            pls_components=pls_components,
            pls_max_components=pls_max_components,
            pls_n_perm=pls_n_perm,
            return_components=return_components,
            n_folds=n_folds,
            seed_inner=seed_inner,
            n_jobs=n_jobs,
            verbose=verbose,
            impute_data=impute_data,
            model_path_template=model_path_template,
            overwrite=overwrite,
        )
        if model_type == "pls"
        else None
    )

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
            pls_n_components=resolved_pls_k,
            pls_max_components=pls_max_components,
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
            if model_type == "pls" and resolved_pls_k is not None:
                cached_k = int(extract_pls_estimator(pipeline).n_components)
                if cached_k != resolved_pls_k:
                    raise ValueError(
                        f"{os.path.basename(model_path)} was fit with {cached_k} PLS component(s), "
                        f"not {resolved_pls_k}. Refit it by setting 'overwrite' to True."
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
    comps_arr = np.full((n_samples, resolved_pls_k), np.nan, dtype=np.float32) if return_components else None
    ref_x_weights: np.ndarray | None = None
    best_sign_flips: np.ndarray | None = None
    fold_abs_r: list[np.ndarray] = []
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

            sign_flips = None
            if return_components:
                W_k = extract_pls_estimator(pipe_k).x_weights_
                if ref_x_weights is None:
                    ref_x_weights = W_k.copy()
                    sign_flips = np.ones(W_k.shape[1], dtype=np.float32)
                else:
                    dots = np.sum(ref_x_weights * W_k, axis=0)
                    norms = np.linalg.norm(ref_x_weights, axis=0) * np.linalg.norm(W_k, axis=0)
                    fold_abs_r.append(np.abs(dots) / np.where(norms > 0, norms, 1.0))
                    sign_flips = np.where(dots < 0, -1.0, 1.0).astype(np.float32)

                comps_arr[idx_tr[va]] = transform_pls_components(pipe_k, _get_features(idx_tr[va])) * sign_flips

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
                best_sign_flips = sign_flips

            if model_perf_path:
                perfs.append({"Split": f"Val_fold-{k}", "N": len(idx_tr[va]), **perf_va})

        if return_components and fold_abs_r:
            mean_abs_r = np.mean(fold_abs_r, axis=0)
            min_abs_r = np.min(fold_abs_r, axis=0)
            summary_str = ", ".join(
                f"PLS{j + 1}={mean_abs_r[j]:.2f}(min {min_abs_r[j]:.2f})"
                for j in range(len(mean_abs_r))
            )
            custom_print(f"\nCross-fold PLS weight |cos| vs fold-0: {summary_str}")

        y_pred[idx_te] = _predict(best_pipe, idx_te)
        if return_components:
            comps_arr[idx_te] = transform_pls_components(best_pipe, _get_features(idx_te)) * best_sign_flips

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

    if return_components:
        comp_cols = [ f"PLS{j + 1}" for j in range(comps_arr.shape[1]) ]
        return y_pred_df, y_pred_ac_df, fold_n, pd.DataFrame(comps_arr, columns=comp_cols)

    return y_pred_df, y_pred_ac_df, fold_n