#!/usr/bin/env python3


import argparse
from pathlib import Path

import pandas as pd

from make_df_scores import SCORE_KEY
from predict_ages import SET, AGE, COG
import calc_corr
from summ_results import TARG_COL, FEAT_COL, load_preds
from plotting import (
    BASE_FS, Style, model_name_2_label, 
    plot_age_scatter, plot_coef_bars, plot_mae_bars, plot_corr_boxes, plot_corr_heat
)

PLOT_OPTIONS = {
    "all"         : "every one of the following", 
    "age_scatters": "predicted versus chronological age, one file per model.",  # from the 'predictions.csv' in the second-level model's result folder
    "coef_bars"   : "weight each first-level model carries in the second-level models",  
    "mae_bars"    : "mean absolute error (MAE) per model, averaged across seeds",  # from the evaluation outputs
    "corr_boxes"  : "distribution of each model's correlations across the scores, one box per model",
    "corr_heat"   : "the same correlations as a model x score matrix"
}

class Config(calc_corr.Config):
    def __init__(self, args: argparse.Namespace):
        self.to_draw = (
            list(PLOT_OPTIONS.keys()) if args.to_draw == 0
            else [ list(PLOT_OPTIONS.keys())[i] for i in set(args.to_draw) ]
        )
        self.style = Style(
            fig_scale=args.fig_scale, 
            font_scale=args.font_scale, 
            dpi=args.dpi
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
        # self.lv2_mdl = lv2_mdl_name.split("_")[0]

        fig_out_dir_1 = self.proj_root / "figures" / lv1_mdl_name / lv2_mdl_name
        fig_out_dir_2 = fig_out_dir_1 / eval_name / corr_name

        self.coefs_path      = lv2_res_dir / "coefficients.csv"
        self.preds_path      = lv2_res_dir / "predictions.csv"
        self.pred_fits_path  = lv2_res_dir / f"pred_fits.csv"
        self.corr_agg_s_path = self.agg_s_out_path  # aggregated correlation statistics over seeds per model x score
        self.corr_agg_m_path = self.agg_m_out_path  # aggregated correlation statistics over seeds per model

        pad_name = "PAD" + (f"-{self.pad_type}" if self.pad_type != "raw" else "")
        self.age_scatter_templ = fig_out_dir_1 / "[scatter] Fits between real and predicted ages ({}).png"
        self.coef_bars_templ   = fig_out_dir_1 / "[bars] Coefficients of the fold-{} 2nd-level model ({}).png"
        self.mae_bars_path     = fig_out_dir_2 / f"[bars] MAE of 2nd-level models ({self.data_set}).png"
        self.corr_boxes_path   = fig_out_dir_2 / f"[boxes] Correlations between {pad_name} and {self.score_name} scores ({self.data_set}).png"
        self.corr_heat_path    = fig_out_dir_2 / f"[heat] Correlations between {pad_name} and {self.score_name} scores ({self.data_set}).png"


def parse_args(argv: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--to_draw", type=int, nargs="+", choices=range(len(PLOT_OPTIONS)), default=0, 
                        help="which figures to draw; " + 
                             ", ".join([ f"{i}: {k} ({v})" for i, (k, v) in enumerate(PLOT_OPTIONS.items()) ]))

    grp_fig = parser.add_argument_group("figure")
    grp_fig.add_argument("--fig_scale", type=float, default=1.,
                         help="Scaling multiplier for the canvas and all its elements")
    grp_fig.add_argument("--font_scale", type=float, default=1.,
                         help=f"Scaling multiplier for all text sizes ({BASE_FS} pt base)")
    grp_fig.add_argument("--dpi", type=int, default=200,
                         help="Resolution the figures are saved at")

    return calc_corr.parse_args(argv, parser=parser)


def feat_to_block(feat: str) -> str:
    '''
    Name of the first-level block a second-level feature came from
    (i.e. the column name with its target prefix removed)
    (e.g., "Age_MRI_ROI" -> "MRI_ROI")
    '''
    target, _, block = feat.partition("_")
    if (target in [AGE, COG]) and (block != ""):
        return block
    else:
        return feat


def main(config: Config):
    if "age_scatters" in config.to_draw:
        assert config.pred_fits_path.is_file(), f"'{config.pred_fits_path}' not exists. You should run `summ_results.py` first."

        preds_df, preds_cols = load_preds(config.preds_path)
        if config.data_set != "all":
            preds_df = preds_df.query(f"{SET} == '{config.data_set}'")

        # age_lo = min(preds_df[AGE].min(), preds_df.loc[:, preds_cols].min().min())
        # age_hi = max(preds_df[AGE].max(), preds_df.loc[:, preds_cols].max().max())
        # pad = (age_lo, age_hi) *.04
        # age_lims = (age_lo - pad, age_hi + pad)
        age_lims = (0, 100)

        fits_df = pd.read_csv(config.pred_fits_path)
        fits_df = fits_df.query(f"Split == '{config.data_set}'")
        fits_df.set_index("Model", inplace=True)  # should be unique

        for preds_col in preds_cols:
            model = preds_col.replace(f"{AGE}_", "")
            plot_age_scatter(
                real_ages=preds_df[AGE].to_numpy(dtype=float),
                pred_ages=preds_df[preds_col].to_numpy(dtype=float), 
                age_lims=age_lims, 
                subj_sets=preds_df[SET].to_list(), 
                subj_set_label=config.data_set, 
                N=fits_df.at[model, "N"], 
                slope=fits_df.at[model, "slope_pred"], 
                intercept=fits_df.at[model, "const_pred"], 
                r=fits_df.at[model, "r_pred"], 
                out_path=Path(str(config.age_scatter_templ).format(model_name_2_label(model))), 
                style=config.style
            )

    if "coef_bars" in config.to_draw:
        assert config.coefs_path.is_file(), f"'{config.coefs_path}' not exists. You should run `summ_results.py` first."
        coefs_df = pd.read_csv(config.coefs_path, index_col=[TARG_COL, FEAT_COL])
        targs = coefs_df.index.get_level_values(TARG_COL).unique()
        feats = coefs_df.index.get_level_values(FEAT_COL).unique()

        for fold_n in coefs_df.columns:
            weights = coefs_df[fold_n].unstack(TARG_COL).reindex(index=feats, columns=targs)
            weights = weights.groupby(feats.map(feat_to_block), sort=False).sum()
            # weights = weights.abs().groupby(feats.map(feat_to_block), sort=False).sum()

            if len(targs) > 1:
                weights["mean"] = weights.mean(axis=1)
                # weights["mean"] = weights.abs().mean(axis=1)

            for col in weights.columns:
                plot_coef_bars(
                    dat=weights[col].sort_values(), 
                    out_path=Path(str(config.coef_bars_templ).format(fold_n, col)), 
                    style=config.style
                )

    if any([ x in config.to_draw for x in ["mae_bars", "corr_boxes", "corr_heat"] ]):
        assert config.corr_agg_m_path.is_file(), f"'{config.corr_agg_m_path}' not exists. You should run `calc_corr.py` first."

        corr_agg_m = pd.read_csv(config.corr_agg_m_path, index_col="Model")
        # corr_agg_m = corr_agg_m.sort_values("MAE")  # already sorted

        corr_agg_s = pd.read_csv(config.corr_agg_s_path)
        # corr_agg_s["Model"] = pd.Categorical(corr_agg_s["Model"], categories=corr_agg_m["Model"], ordered=True)
        # corr_agg_s = corr_agg_s.sort_values(["Model", "Score"]).reset_index(drop=True)

        if "mae_bars" in config.to_draw:        
            plot_mae_bars(
                dat=corr_agg_m["MAE"], 
                dat_sd=corr_agg_m["MAE_SD"], 
                out_path=config.mae_bars_path, 
                style=config.style
            )

        if "corr_boxes" in config.to_draw:
            plot_corr_boxes(
                df=corr_agg_s, 
                model_order=corr_agg_m.index.to_list(), 
                model_stars=corr_agg_m["Stars"], 
                fdr_alpha=config.fdr_alpha, 
                out_path=config.corr_boxes_path, 
                style=config.style
            )

        if "corr_heat" in config.to_draw:
            # cog_data = json.loads(config.scores_json_path.read_text())
            # score_key = cog_data["score_key"]
            plot_corr_heat(
                df=corr_agg_s, 
                model_order=corr_agg_m.index.to_list(), 
                score_key=SCORE_KEY, 
                fdr_alpha=config.fdr_alpha, 
                out_path=config.corr_heat_path, 
                style=config.style
            )


if __name__ == "__main__":
    main(Config(parse_args()))