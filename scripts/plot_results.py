#!/usr/bin/env python3


import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import calc_corr
from make_df_scores import SCORE_KEY
from predict_ages import SET, AGE, COG, MIX
from summ_results import TARG_COL, FEAT_COL, TREE_TARG_NAME, load_preds
from plotting import (
    BASE_FS, Style, model_name_2_label,
    plot_age_scatter, plot_coef_bars, plot_mae_bars, plot_corr_boxes, plot_corr_heat,
)


PLOT_OPTIONS = {
    "all"         : "every one of the following",
    "age_scatters": "predicted versus chronological age, one file per model",
    "coef_bars"   : "weight each first-level model carries in the second-level models",
    "mae_bars"    : "mean absolute error (MAE) per model, averaged across seeds",
    "corr_boxes"  : "distribution of each model's correlations across the scores, one box per model",
    "corr_heat"   : "the same correlations as a model x score matrix",
}
PLOT_KEYS = list(PLOT_OPTIONS.keys())
INDIVIDUAL_PLOTS = [ k for k in PLOT_KEYS if k != "all" ]


class Config(calc_corr.Config):
    def __init__(self, args: argparse.Namespace):
        selected = set(args.to_draw)
        self.to_draw = (
            list(INDIVIDUAL_PLOTS) if 0 in selected
            else [ PLOT_KEYS[i] for i in sorted(selected) ]
        )
        self.style = Style(
            fig_scale=args.fig_scale,
            font_scale=args.font_scale,
            dpi=args.dpi,
        )
        super().__init__(args)  # calls `setup_vars` and `setup_paths`

    def setup_paths(self, args):
        self.proj_root = Path(__file__).resolve().parents[1]
        super().setup_paths(args)

        corr_name = self.out_dir.name  # inherited
        eval_name = self.eval_dir.name  # inherited
        lv2_res_dir = self.eval_dir.parent
        lv2_mdl_name = lv2_res_dir.name
        lv1_mdl_name = lv2_res_dir.parent.name

        fig_out_dir_1 = self.proj_root / "figures" / lv1_mdl_name / lv2_mdl_name
        fig_out_dir_2 = fig_out_dir_1 / eval_name / corr_name

        self.coefs_path      = lv2_res_dir / "coefficients.csv"
        self.preds_path      = lv2_res_dir / "predictions.csv"
        self.pred_fits_path  = lv2_res_dir / "pred_fits.csv"
        self.corr_agg_s_path = self.agg_s_out_path  # aggregated correlation stats over seeds per model x score
        self.corr_agg_m_path = self.agg_m_out_path  # aggregated correlation stats over seeds per model

        pad_name = "PAD" + (f"-{self.pad_type}" if self.pad_type != "raw" else "")
        self.age_scatter_templ = fig_out_dir_1 / "[scatter] Fits between real and predicted ages ({}).png"
        self.coef_bars_templ   = fig_out_dir_1 / "[bars] Coefficients of the fold-{} 2nd-level model ({}).png"
        self.norm_bars_templ   = fig_out_dir_1 / "[bars] Weight norms (L2) of the fold-{} 2nd-level model ({}).png"
        self.imp_bars_templ    = fig_out_dir_1 / "[bars] Importances of the fold-{} 2nd-level model ({}).png"
        self.mae_bars_path     = fig_out_dir_2 / f"[bars] MAE of 2nd-level models ({self.data_set}).png"
        self.corr_boxes_path   = fig_out_dir_2 / f"[boxes] Correlations between {pad_name} and {self.score_name} scores ({self.data_set}).png"
        self.corr_heat_path    = fig_out_dir_2 / f"[heat] Correlations between {pad_name} and {self.score_name} scores ({self.data_set}).png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot evaluation results for the two-level age-prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--to_draw", type=int, nargs="+", choices=range(len(PLOT_OPTIONS)), default=[0], 
                        help=("which figures to draw; " + 
                              ", ".join(f"{i}: {k} ({v})" for i, (k, v) in enumerate(PLOT_OPTIONS.items()))))

    grp_fig = parser.add_argument_group("figure")
    grp_fig.add_argument("--fig_scale", type=float, default=1.0,
                         help="Scaling multiplier for the canvas and all its elements")
    grp_fig.add_argument("--font_scale", type=float, default=1.0,
                         help=f"Scaling multiplier for all text sizes ({BASE_FS} pt base)")
    grp_fig.add_argument("--dpi", type=int, default=200,
                         help="Resolution the figures are saved at")

    return calc_corr.parse_args(argv, parser=parser)


def feat_to_block(feat: str) -> str:
    '''
    Name of the first-level block a second-level feature came from:
    - "MRI_ROI_PLS12" -> "MRI_ROI"
    - "Age_MRI_ROI" / "Cog_DTI_FA" / "Mix_Pyment" -> "MRI_ROI" / "DTI_FA" / "Pyment"
    '''
    base = re.sub(r"_PLS\d+$", "", feat)
    if base != feat:
        return base

    target, _, block = feat.partition("_")
    if (target in {AGE, COG, MIX}) and block:
        return block

    return feat


def main(config: Config) -> None:
    if "age_scatters" in config.to_draw:
        if not config.pred_fits_path.is_file():
            raise FileNotFoundError(f"'{config.pred_fits_path}' does not exist. Run `summ_results.py` first.")

        preds_df, preds_cols = load_preds(config.preds_path)
        if config.data_set != "all":
            preds_df = preds_df[preds_df[SET] == config.data_set]

        age_lims = (0, 100)
        fits_df = pd.read_csv(config.pred_fits_path)
        fits_df = fits_df[fits_df["Split"] == config.data_set].set_index("Model")

        real_ages = preds_df[AGE].to_numpy(dtype=float)
        subj_sets = preds_df[SET].to_list()

        for preds_col in preds_cols:
            model = preds_col.removeprefix(f"{AGE}_")
            if model not in fits_df.index:
                continue

            fit_row = fits_df.loc[model]
            plot_age_scatter(
                real_ages=real_ages,
                pred_ages=preds_df[preds_col].to_numpy(dtype=float),
                age_lims=age_lims,
                subj_sets=subj_sets,
                subj_set_label=config.data_set,
                N=int(fit_row["N"]),
                slope=float(fit_row["slope_pred"]),
                intercept=float(fit_row["const_pred"]),
                r=float(fit_row["r_pred"]),
                out_path=Path(str(config.age_scatter_templ).format(model_name_2_label(model))),
                style=config.style,
            )

    if "coef_bars" in config.to_draw:
        if not config.coefs_path.is_file():
            raise FileNotFoundError(f"'{config.coefs_path}' does not exist. Run `summ_results.py` first.")

        coefs_df = pd.read_csv(config.coefs_path, index_col=[TARG_COL, FEAT_COL])
        targs = coefs_df.index.get_level_values(TARG_COL).unique()
        feats = coefs_df.index.get_level_values(FEAT_COL).unique()
        block_keys = feats.map(feat_to_block)
        is_tree_imp = list(targs) == [TREE_TARG_NAME]
        is_pls_feats = bool(feats.str.contains(r"_PLS\d+$").any())
        use_l2_norm = is_pls_feats and not is_tree_imp

        bar_templ = (
            config.imp_bars_templ if is_tree_imp
            else config.norm_bars_templ if use_l2_norm
            else config.coef_bars_templ
        )

        for fold_n in coefs_df.columns:
            weights = coefs_df[fold_n].unstack(TARG_COL).reindex(index=feats, columns=targs)
            if use_l2_norm:
                # PLS component signs are arbitrary; L2 norm keeps multi-component blocks comparable to 1-col Pyment
                weights = np.sqrt((weights ** 2).groupby(block_keys, sort=False).sum())
            else:
                # Linear target predictions (signed) or tree importances (non-negative, sum to 1)
                weights = weights.groupby(block_keys, sort=False).sum()

            if len(targs) > 1:
                weights["mean"] = weights.mean(axis=1)

            for col in weights.columns:
                plot_coef_bars(
                    dat=weights[col].sort_values(),
                    out_path=Path(str(bar_templ).format(fold_n, col)),
                    style=config.style,
                )

    if any(x in config.to_draw for x in ["mae_bars", "corr_boxes", "corr_heat"]):
        if not config.corr_agg_m_path.is_file():
            raise FileNotFoundError(f"'{config.corr_agg_m_path}' does not exist. Run `calc_corr.py` first.")

        corr_agg_m = pd.read_csv(config.corr_agg_m_path, index_col="Model")
        corr_agg_s = pd.read_csv(config.corr_agg_s_path)
        model_order = corr_agg_m.index.to_list()

        if "mae_bars" in config.to_draw:
            plot_mae_bars(
                dat=corr_agg_m["MAE"],
                dat_sd=corr_agg_m["MAE_SD"],
                out_path=config.mae_bars_path,
                style=config.style,
            )

        if "corr_boxes" in config.to_draw:
            plot_corr_boxes(
                df=corr_agg_s,
                model_order=model_order,
                model_stars=corr_agg_m["Stars"],
                fdr_alpha=config.fdr_alpha,
                out_path=config.corr_boxes_path,
                style=config.style,
            )

        if "corr_heat" in config.to_draw:
            plot_corr_heat(
                df=corr_agg_s,
                model_order=model_order,
                score_key=SCORE_KEY,
                fdr_alpha=config.fdr_alpha,
                out_path=config.corr_heat_path,
                style=config.style,
            )


if __name__ == "__main__":
    main(Config(parse_args()))