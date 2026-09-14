#!/usr/bin/env python3


import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from predict_ages import SID, SET, AGE


TARG_COL, FEAT_COL = "Target", "Feature"


class Config:
    def __init__(self, args: argparse.Namespace):
        self.proj_root = Path(__file__).resolve().parents[1]

        self.preds_path = Path(args.preds_path) if args.preds_path else self._latest_preds_path()
        lv2_res_dir = self.preds_path.parent
        self.fits_out_path = lv2_res_dir / f"pred_fits.csv"

        lv2_mdl_name = lv2_res_dir.name
        lv1_mdl_name = lv2_res_dir.parent.name
        lv2_mdl_dir = self.proj_root / "models" / lv1_mdl_name / lv2_mdl_name
        self.lv2_model_paths = sorted(lv2_mdl_dir.glob("pipeline_*.joblib"))
        assert self.lv2_model_paths, f"\nNo second-level model found. Did you change file name(s)?"
        self.coefs_out_path = lv2_res_dir / "coefficients.csv"

    def _latest_preds_path(self) -> Path:
        '''
        Most recent second-level model's prediction table
        '''
        pattern = os.path.join("results", "*", "*", "predictions.csv")
        found = sorted(self.proj_root.glob(pattern))
        assert found, f"\nNo {pattern} found; pass --preds_path\n"
        return found[-1]


def parse_args(argv: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-p", "--preds_path", default=None, 
                        help="Path to the prediction.csv the scatter plot will be drawn from" + 
                             "; also determines path to the second-level models whose weights will be read"
                             "; None picks the latest one")

    return parser.parse_args(argv)


def load_preds(preds_path: Path) -> tuple[pd.DataFrame, list[str]]:
    '''
    Read and return the prediction table, and
    the column names of the predictions from the evaluated models (Age_*; to collect model names).
    No selection applied.
    '''
    df = pd.read_csv(preds_path, index_col=SID)
    cols = [ c for c in df.columns if c.startswith(f"{AGE}_") ]
    
    print(f"\nLoaded predictions from folder: {preds_path.parent}")
    print(f"{len(cols)} model(s), {len(df)} participant(s).\n")

    return df, cols


def calc_fits(preds_df: pd.DataFrame, preds_cols: list[str]) -> pd.DataFrame:
    '''
    Least-squares fit (constant and slope) and correlation coefficient
    of the predicted ages on the chronological ages, 
    calculated separately for each data set/split and model.
    '''
    rows = []

    for split in ["all", "test", "train"]:
        sub_df = preds_df if split == "all" else preds_df[preds_df[SET] == split]

        if not len(sub_df):
            continue

        y_true_full = sub_df[AGE].to_numpy(dtype=float)

        for p_col in preds_cols:
            model = p_col.replace(f"{AGE}_", "") if "Final" not in p_col else "Final"
            y_pred_full = sub_df[p_col]

            mask = ~np.isnan(y_pred_full)
            y_true = y_true_full[mask]
            y_pred = y_pred_full[mask]

            coefs_pred = np.polyfit(y_true, y_pred, 1)
            r_pred = np.corrcoef(y_true, y_pred)[0, 1]

            pad = y_pred - y_true
            coefs_pad = np.polyfit(y_true, pad, 1)
            r_pad = np.corrcoef(y_true, pad)[0, 1]

            mae = float(np.nanmean(np.abs(pad)))
            r2 = 1 - np.sum(pad ** 2) / np.sum((y_true - y_true.mean()) ** 2)

            rows.append({
                "Model": model, 
                "Split": split,
                "N": len(y_pred),
                "const_pred": coefs_pred[1],  # intercept
                "slope_pred": coefs_pred[0],
                "r_pred": r_pred, 
                "const_pad": coefs_pad[1], 
                "slope_pad": coefs_pad[0],
                "r_pad": r_pad, 
                "MAE": mae, 
                "R2": r2
            })

    return pd.DataFrame(rows)


def load_coefs(lv2_model_paths: list[Path]) -> pd.DataFrame:
    '''
    Read the weights of the second-level models from each fold.
    '''
    F = None
    coefs_dict = {}

    for path in lv2_model_paths:  # pipline fit on each fold
        fold_n = int(path.stem.split("-")[-1])

        pipe = joblib.load(path)
        model = pipe.named_steps["model"]
        model = getattr(model, "regressor_", model)  # for TransformedTargetRegressor

        coefs = np.atleast_2d(model.coef_)
        targets = list(getattr(pipe, "target_names_in_", []))
        
        feats = list(getattr(pipe, "feature_names_in_", []))
        assert F is None or set(feats) == set(F), f"\nFeature set in fold-{fold_n} differ from the other cycles\n"
        F = feats
        
        assert coefs.shape == (len(targets), len(feats)), (
            f"\nThe pipeline names {len(targets)} target(s) and {len(feats)} feature(s), "
            f"but carries a {coefs.shape} weight matrix\n"
        )

        coefs_dict[fold_n] = pd.Series(
            coefs.ravel(), 
            index=pd.MultiIndex.from_product([targets, feats], names=[TARG_COL, FEAT_COL])
        )

    return pd.DataFrame.from_dict(coefs_dict)


def main(config: Config):
    preds_df, preds_cols = load_preds(config.preds_path)
    fits_df = calc_fits(preds_df, preds_cols)
    fits_df.to_csv(config.fits_out_path, index=False)
    print(f"Saved: {config.fits_out_path}\n")

    coefs_df = load_coefs(config.lv2_model_paths)
    coefs_df.to_csv(config.coefs_out_path)
    print(f"Saved: {config.coefs_out_path}\n")


if __name__ == "__main__":
    main(Config(parse_args()))