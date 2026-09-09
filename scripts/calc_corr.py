#!/usr/bin/env python3


import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests

from plotting import sig_stars


DATA_SETS = ["all", "test", "train"]
METHODS = ["pearson", "spearman"]  # any valid option for: https://pingouin-stats.org/generated/pingouin.partial_corr.html
PAD_OPTIONS = { 
    "ac" : "use age-corrected predictions ('C-Age_*'), so the effect of age is already taken into account",
    "pa" : "use the original predictions ('Age_*'), with age as a covariate when calculating the correlation(s)",
    "raw": "just use the the original predictions"
}
FDR_SCOPES = ["figure", "model"]  # the family the point-level p values are corrected within


class Config:
    def __init__(self, args: argparse.Namespace):
        self.setup_vars(args)
        self.setup_paths(args)
        self.overwrite = args.overwrite

    def setup_vars(self, args):
        self.score_name = "ST"
        self.score_key = "_ST_SCALED_"
        self.data_set = DATA_SETS[args.data_set]
        self.method = METHODS[args.method]
        self.pad_type = list(PAD_OPTIONS.keys())[args.pad_type]
        self.partial_score = args.partial_score
        self.fdr_alpha = args.fdr_alpha
        self.fdr_scope = FDR_SCOPES[args.fdr_scope]

    def setup_paths(self, args):
        self.proj_root = Path(__file__).resolve().parents[1]
        self.full_tbl_path = self.proj_root / "data" / "tabular" / "df_merged.csv"

        self.eval_dir = Path(args.eval_dir) if args.eval_dir else self._latest_eval_dir()
        # a folder that is not there and one that holds no tables fail differently: the
        # first is a mistyped path (or an unset shell variable), the second a run that
        # has not been evaluated yet
        assert self.eval_dir.is_dir(), f"\nNo such folder: {self.eval_dir.resolve()}\n"

        self.preds_paths = sorted(self.eval_dir.glob("predictions_seed-*.csv"))
        assert self.preds_paths, f"\nNo 'predictions_seed-*.csv' under {self.eval_dir}\n"

        pad_name = f"pad-{self.pad_type}" if self.pad_type != "raw" else "pad"
        score_name = f"{self.score_name}-pa" if self.partial_score else self.score_name
        self.out_dir = self.eval_dir / f"{pad_name}_x_{score_name} ({self.method})"
        self.long_out_path  = self.out_dir / f"long_{self.data_set}.csv"
        self.mae_summ_path  = self.out_dir / f"mae_{self.data_set}.csv"
        self.agg_s_out_path = self.out_dir / f"by-score_{self.data_set}_{self.fdr_scope}-{self.fdr_alpha}.csv"
        self.agg_m_out_path = self.out_dir / f"by-model_{self.data_set}_{self.fdr_scope}-{self.fdr_alpha}.csv"

    def _latest_eval_dir(self) -> Path:
        '''
        Most recent feature-evaluation folder that carries prediction tables
        '''
        eval_preds_pattern = os.path.join("results", "*", "*", "eval_*", "predictions_seed-*.csv")
        eval_dir_found = sorted(set( p.parent for p in self.proj_root.glob(eval_preds_pattern) ))
        assert eval_dir_found, f"\nNo {eval_preds_pattern} found; pass --eval_dir\n"
        return eval_dir_found[-1]


def parse_args(
    argv: list[str] = None, 
    parser: argparse.ArgumentParser = None, 
    defaults: dict = None
) -> argparse.Namespace:
    parser = parser or argparse.ArgumentParser(
        description="Correlate PAD values with scores from the merged table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="overwrite existing outputs")

    grp_data = parser.add_argument_group("data")
    grp_data.add_argument("-d", "--eval_dir", default=None,
                          help="folder holding the 'predictions_seed-*.csv' tables; None picks the latest one")
    grp_data.add_argument("-set", "--data_set", type=int, choices=range(len(DATA_SETS)), default=0, 
                          help="which participants to correlate over; " + 
                               ", ".join( f"'{i}': {x}" for i, x in enumerate(DATA_SETS) ))
    
    grp_corr = parser.add_argument_group("stats")
    grp_corr.add_argument("-m", "--method", type=int, choices=range(len(METHODS)), default=0, 
                          help="which correlation measure to use; " +
                               ", ".join( f"'{i}': {x}" for i, x in enumerate(METHODS) ))
    grp_corr.add_argument("-pad", "--pad_type", type=int, choices=range(len(PAD_OPTIONS)), default=1, 
                          help="whether to remove age-related effects from PAD; " +
                               ", ".join( f"'{i}': {v}" for i, v in enumerate(PAD_OPTIONS.values()) ))
    grp_corr.add_argument("-ps", "--partial_score", action="store_true",
                          help="also residualize the scores on chronological age")
    grp_corr.add_argument("-fdr", "--fdr_alpha", type=float, default=.05,
                          help="FDR-corrected p below which a correlation counts as significant")
    grp_corr.add_argument("-fs", "--fdr_scope", type=int, choices=range(len(FDR_SCOPES)), default=0, 
                          help="correct the point-level p values across every model at once (0), or within each model (1)")

    parser.set_defaults(**(defaults or {}))

    return parser.parse_args(argv)


def load_preds(config: Config) -> tuple[dict[int, pd.DataFrame], list[str], list[str]]:
    '''
    Read and return the prediction table (no column selection applied) per seed, 
    the column names of the predictions from the evaluated models (Age_*; to collect model names),
    and the ID of the participants (selected by `config.data_set`).
    '''
    cols, sids = None, None
    df_dict = {}

    for path in config.preds_paths:
        df = pd.read_csv(path, index_col="SID")
        preds_cols = [ c for c in df.columns if c.startswith("Age_") ]

        assert cols is None or preds_cols == cols, f"\nModel columns of {path.name} differ from the other tables\n"
        cols = preds_cols

        if config.pad_type == "ac":
            missing = [ c for c in preds_cols if f"C-{c}" not in df.columns ]
            assert not missing, f"\nChoose to use age-corrected PAD, but {path.name} carries no 'C-' column for {len(missing)} model(s)."

        if config.data_set != "all":
            df = df[df["Set"] == config.data_set]

        assert sids is None or df.index.equals(sids), f"\nParticipants of {path.name} differ from the other tables\n"
        sids = df.index

        seed = int(path.stem.split("-")[-1])
        df_dict[seed] = df

    sids = sids.to_list()
    print(f"\n{len(df_dict)} seed(s) from: {config.eval_dir}")
    print(f"{len(cols)} model(s), {len(sids)} participant(s) in the '{config.data_set}' set")

    return df_dict, cols, sids


def load_scores(sids: list[str], config: Config) -> pd.DataFrame:
    '''
    Read the columns containing the string `config.score_key` 
    from the file `config.full_tbl_path`, 
    and return the data of the participants specified with `sid`.
    '''
    cols = pd.read_csv(config.full_tbl_path, nrows=0).columns
    sel_cols = [ c for c in cols if config.score_key in c ]
    assert sel_cols, f"\nNo column of {config.full_tbl_path.name} holds '{config.score_key}'\n"
    print(f"\n{len(sel_cols)} columns(s) contains the string '{config.score_key}'")

    print(f"Loading matched columns(s) from: {config.full_tbl_path}")
    df = pd.read_csv(config.full_tbl_path, index_col="BASIC_INFO_ID", usecols=["BASIC_INFO_ID", "BASIC_INFO_AGE"] + sel_cols)
    df = df.rename(columns={"BASIC_INFO_AGE": "Age"})

    sel_df = df.loc[sids, ["Age"] + sel_cols]
    sel_df = sel_df.apply(pd.to_numeric, errors="coerce")

    missing = int(sel_df[sel_cols].isna().all(axis=1).sum())
    print(f"{missing} participant(s) without any score\n")

    return sel_df


def calc_corr(
    preds_df_dict: dict[int, pd.DataFrame], 
    preds_cols: list[str], 
    score_df: pd.DataFrame, 
    method: str, 
    pad_type: str, 
    partial_score: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Correlate every model's prediction errors (PAD) with every score, once per seed.
    Returns the correlations and the models' MAEs summarized per seed.
    '''
    corr_rows, mae_rows = [], []

    age = score_df["Age"]
    score_cols = list(score_df.columns)
    score_cols.remove("Age")

    covars = (
        {"covar": "Age"} if (pad_type == "pa") and partial_score
        else {"y_covar": "Age"} if pad_type == "pa"
        else {"x_covar": "Age"} if partial_score
        else {}
    )

    for seed, preds_df in preds_df_dict.items():
        for p_col in preds_cols:
            err = preds_df[p_col] - age
            model = p_col.replace("Age_", "")
            mae_rows.append({
                "Seed": seed, 
                "Model": model, 
                "MAE": float(np.nanmean(np.abs(err)))
            })
            pad = preds_df[f"C-{p_col}"] - age if pad_type == "ac" else err

            for s_col in score_cols:
                stat_out = pg.partial_corr(
                    data=pd.DataFrame({
                        s_col: score_df[s_col], 
                        model: pad, 
                        "Age": age
                    }), 
                    x=s_col, 
                    y=model, 
                    **covars, 
                    method=method
                )
                corr_rows.append({
                    "Seed"  : seed,
                    "Model" : model,
                    "Score" : s_col,
                    "Domain": s_col.split("_")[0],
                    "Scale" : s_col.split("_")[2],  # RAW / SCALED / NORM
                    **stat_out.iloc[0].to_dict()  # `pg.partial_corr` hands back a one-row frame
                })

    return pd.DataFrame(corr_rows), pd.DataFrame(mae_rows)


def fisher_mean(r_vals: pd.Series) -> float:
    '''
    Averaging correlations leads to underestimation 
    because the sampling distribution of the correlation coefficient (r) is skewed, 
    particularly when sample size is small.    

    It was found that transforming r by Fisher's z prior to averaging, 
    and then back-transforming the averaged z to r, is less biased.

    - see: https://doi.org/10.1037/0021-9010.72.1.146
    '''
    z_vals = np.arctanh(np.clip(r_vals, -1 + 1e-7, 1 - 1e-7))
    mean_z = np.nanmean(z_vals)
    mean_r = np.tanh(mean_z)

    return mean_r


def convert_p_to_q(p: pd.Series) -> pd.Series:
    '''
    Controlling false discovery rate (FDR) with Benjamini-Hochberg (BH) procedure, 
    which converts p-values to q-values.
    '''
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def agg_stats_per_score(corr_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    '''
    Aggregate correlation statistics over the seeds per (model x score)
    '''
    df = corr_df.groupby(["Model", "Score", "Domain", "Scale"], as_index=False).agg(
        r=("r", fisher_mean), 
        r_SD=("r", lambda r: float(np.std(r, ddof=1)) if len(r) > 1 else np.nan),
        N=("n", "mean"),
        N_seeds=("r", "size")
    )
    mean_r = df["r"]
    n = df["N"]
    dof = n - 2 - int(config.pad_type == "pa" or config.partial_score)  # whether age was partialled out of one side or of both, it costs one dof
    resid = np.clip(1 - mean_r ** 2, 1e-12, None)
    t = mean_r * np.sqrt(dof / resid)
    mean_p = 2 * stats.t.sf(np.abs(t), dof)  # survival function, two-tailed
    df["p"] = mean_p

    if config.fdr_scope == "model":
        df["q"] = df.groupby("Model", observed=True)["p"].transform(convert_p_to_q)
    else:
        df["q"] = convert_p_to_q(df["p"])

    df["q_sig"] = df["q"] < config.fdr_alpha

    print(f"At point level, after correction for FDR < {config.fdr_alpha:.2f} (scope: {config.fdr_scope}), "
          f"{int(df['q_sig'].sum())} out of {df['p'].notna().sum()} correlation(s) are significant\n")

    return df


def agg_stats_per_model(agg_by_score: pd.DataFrame, mae_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    '''
    Aggregate correlation statistics over the seeds per model
    '''
    def _wilcoxon(r_vals: pd.Series, which: int) -> float:
        r_vals = r_vals.dropna().to_numpy()
        return float(stats.wilcoxon(r_vals, alternative="two-sided")[which])  # hypothesized median = 0

    df = agg_by_score.groupby("Model", as_index=False).agg(
        N_scores=("r", "size"),
        r_mean=("r", fisher_mean),
        r_absmean=("r", lambda r: float(np.mean(np.abs(r)))),
        r_min=("r", "min"), 
        r_median=("r", "median"), 
        r_max=("r", "max"), 
        N_sig=("q_sig", "sum"),  # number of significant correlations at the point level
        W=("r", lambda r: _wilcoxon(r, 0)),
        p_W=("r", lambda r: _wilcoxon(r, 1)),
    )
    df["q_W"] = df["p_W"].transform(convert_p_to_q)
    df["Stars"] = df["q_W"].map(lambda q: sig_stars(q, config.fdr_alpha))

    df = pd.merge(
        mae_df.groupby("Model", as_index=False).agg(
            MAE=("MAE", "mean"),
            MAE_SD=("MAE", lambda m: float(np.std(m, ddof=1)) if len(m) > 1 else np.nan)
        ), 
        df, 
        on="Model"
    ).sort_values("MAE").reset_index(drop=True)

    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    N_q_sig = int((df["q_W"] < config.fdr_alpha).sum())
    print(f"After correction for FDR < {config.fdr_alpha:.2f}, "
          f"{N_q_sig} out of {len(df)} model(s) are significant\n")

    return df


def main(config: Config):
    config.out_dir.mkdir(parents=True, exist_ok=True)

    if not config.overwrite and config.agg_s_out_path.is_file() and config.agg_m_out_path.is_file():
        print("\nCorrelations stats exists, not overwriting ...\n")
        # agg_by_score = pd.read_csv(config.agg_s_out_path)
        # agg_by_model = pd.read_csv(config.agg_m_out_path)
    else:
        if not config.overwrite and config.long_out_path.is_file() and config.mae_summ_path.is_file():
            corr_df = pd.read_csv(config.long_out_path)
            mae_df = pd.read_csv(config.mae_summ_path)
        else:
            preds_df_dict, preds_cols, sids = load_preds(config)
            score_df = load_scores(sids, config)
            corr_df, mae_df = calc_corr(
                preds_df_dict, preds_cols, score_df, config.method, config.pad_type, config.partial_score
            )
            corr_df.to_csv(config.long_out_path, index=False)
            mae_df.to_csv(config.mae_summ_path, index=False)

        agg_by_score = agg_stats_per_score(corr_df, config)
        agg_by_model = agg_stats_per_model(agg_by_score, mae_df, config)

        agg_by_score["Model"] = pd.Categorical(agg_by_score["Model"], categories=agg_by_model["Model"], ordered=True)
        agg_by_score = agg_by_score.sort_values(["Model", "Score"]).reset_index(drop=True)

        for df, path in [
            (agg_by_score, config.agg_s_out_path), 
            (agg_by_model, config.agg_m_out_path)
        ]:
            df.to_csv(path, index=False)
            print(f"Saved: {path}\n")

        shown = ["Rank", "Model", "MAE", "N_scores", "N_sig", "r_mean", "r_median", "r_min", "r_max", "W", "p_W", "Stars"]
        print(agg_by_model[shown].to_string(index=False, float_format="%.3f"))
        print()


if __name__ == "__main__":
    main(Config(parse_args()))