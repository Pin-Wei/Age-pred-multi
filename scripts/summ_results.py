#!/usr/bin/env python3


import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from helpers import get_latest_results
from predict_ages import SID, SET, AGE


TARG_COL, FEAT_COL = "Target", "Feature"
TREE_TARG_NAME = "Importance"


class Config:
    def __init__(self, args: argparse.Namespace):
        self.proj_root = Path(__file__).resolve().parents[1]

        self.preds_path = (
            Path(args.preds_path) if args.preds_path
            else get_latest_results(search_level="lv2", return_level="path")
        )
        if not self.preds_path.is_file():
            raise FileNotFoundError(f"Predictions file not found: {self.preds_path}")

        lv2_res_dir = self.preds_path.parent
        self.fits_out_path  = lv2_res_dir / "pred_fits.csv"
        self.coefs_out_path = lv2_res_dir / "coefficients.csv"

        lv2_mdl_name = lv2_res_dir.name
        lv1_mdl_name = lv2_res_dir.parent.name
        lv2_mdl_dir = self.proj_root / "models" / lv1_mdl_name / lv2_mdl_name
        self.lv2_model_paths = sorted(lv2_mdl_dir.glob("pipeline_*.joblib"))
        if not self.lv2_model_paths:
            raise FileNotFoundError(f"No second-level model found under {lv2_mdl_dir}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize second-level age-prediction fits and feature weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--preds_path", default=None, 
                        help=("Path to the predictions.csv the scatter plot will be drawn from; " + 
                              "also locates the second-level models whose weights will be read. None picks the latest one"))

    return parser.parse_args(argv)


def load_preds(preds_path: Path) -> tuple[pd.DataFrame, list[str]]:
    '''
    Read and return the prediction table and the evaluated model columns (`Age_*`).
    No row selection is applied.
    '''
    df = pd.read_csv(preds_path, index_col=SID)
    cols = [ c for c in df.columns if c.startswith(f"{AGE}_") ]

    print(f"\nLoaded predictions from folder: {preds_path.parent}")
    print(f"{len(cols)} model(s), {len(df)} participant(s).\n")

    return df, cols


def calc_fits(preds_df: pd.DataFrame, preds_cols: list[str]) -> pd.DataFrame:
    '''
    Least-squares fit (constant and slope), correlation coefficient, MAE, and R^2
    of predicted ages on chronological ages, per split and model.
    '''
    rows = []

    for split in ["all", "test", "train"]:
        sub_df = preds_df if split == "all" else preds_df[preds_df[SET] == split]
        if sub_df.empty:
            continue

        y_true_full = sub_df[AGE].to_numpy(dtype=np.float64)

        for p_col in preds_cols:
            model = p_col.removeprefix(f"{AGE}_")
            y_pred_full = sub_df[p_col].to_numpy(dtype=np.float64)

            mask = np.isfinite(y_true_full) & np.isfinite(y_pred_full)
            y_true = y_true_full[mask]
            y_pred = y_pred_full[mask]
            if len(y_true) < 2:
                continue

            slope_pred, const_pred = np.polyfit(y_true, y_pred, 1)
            r_pred = float(np.corrcoef(y_true, y_pred)[0, 1])

            pad = y_pred - y_true
            slope_pad, const_pad = np.polyfit(y_true, pad, 1)
            r_pad = float(np.corrcoef(y_true, pad)[0, 1])

            mae = float(np.mean(np.abs(pad)))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            r2 = float(1.0 - np.sum(pad ** 2) / ss_tot) if ss_tot > 0 else np.nan

            rows.append({
                "Model"     : model,
                "Split"     : split,
                "N"         : len(y_pred),
                "const_pred": float(const_pred),
                "slope_pred": float(slope_pred),
                "r_pred"    : r_pred,
                "const_pad" : float(const_pad),
                "slope_pad" : float(slope_pad),
                "r_pad"     : r_pad,
                "MAE"       : mae,
                "R2"        : r2,
            })

    return pd.DataFrame(rows)


def load_coefs(lv2_model_paths: list[Path]) -> pd.DataFrame | None:
    '''
    Read the weights (or feature importances) of the second-level models from each fold.
    '''
    ref_feats = None
    coefs_dict = {}

    for path in lv2_model_paths:
        suffix = path.stem.removeprefix("pipeline_")
        fold_key: int | str = int(suffix.split("-")[-1]) if suffix.startswith("fold-") else suffix

        pipe = joblib.load(path)
        model = pipe.named_steps["model"]
        model = getattr(model, "regressor_", model)       # for TransformedTargetRegressor
        model = getattr(model, "best_estimator_", model)  # for GridSearchCV

        if hasattr(model, "coef_"):
            coefs = np.atleast_2d(model.coef_)
            targets = list(getattr(pipe, "target_names_in_", []))
        elif hasattr(model, "feature_importances_"):
            coefs = np.atleast_2d(model.feature_importances_)
            targets = [TREE_TARG_NAME]
        else:
            print(f"\n{path.name} holds a {type(model).__name__}, which carries neither coef_ nor feature_importances_\n")
            return None

        feats = list(getattr(pipe, "feature_names_in_", []))
        if ref_feats is not None and set(feats) != set(ref_feats):
            raise ValueError(f"Feature set in '{suffix}' differs from the other folds.")
        ref_feats = feats

        if coefs.shape == (len(feats), len(targets)) and len(targets) != len(feats):
            coefs = coefs.T

        if coefs.shape != (len(targets), len(feats)):
            raise ValueError(
                f"Pipeline '{path.name}' names {len(targets)} target(s) and {len(feats)} feature(s), "
                f"but carries a {coefs.shape} weight matrix."
            )

        coefs_dict[fold_key] = pd.Series(
            coefs.ravel(),
            index=pd.MultiIndex.from_product([targets, feats], names=[TARG_COL, FEAT_COL]),
        )

    return pd.DataFrame.from_dict(coefs_dict)


def main(config: Config) -> None:
    preds_df, preds_cols = load_preds(config.preds_path)
    fits_df = calc_fits(preds_df, preds_cols)
    fits_df.to_csv(config.fits_out_path, index=False)
    print(f"Saved: {config.fits_out_path}\n")

    coefs_df = load_coefs(config.lv2_model_paths)
    if coefs_df is not None:
        coefs_df.to_csv(config.coefs_out_path)
        print(f"Saved: {config.coefs_out_path}\n")


if __name__ == "__main__":
    main(Config(parse_args()))