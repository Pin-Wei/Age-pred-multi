#!/usr/bin/env python3


import argparse
import json
from pathlib import Path

import pandas as pd

from helpers import print_missing
from utils import to_json_compatible


SCORE_NAME = "ST"
SCORE_KEY = "_ST_SCALED_"
COG_SCORE_COLS = [  # one per cognitive domain
    "LANGUAGE_ST_SCALED_SIMILARITY",
    "MEMORY_ST_SCALED_LogMemI",
    "MOTOR_ST_SCALED_ProcessingSpeed"
]


class Config:
    def __init__(self, args: argparse.Namespace):
        self.score_name = SCORE_NAME
        self.score_key = SCORE_KEY
        self.sel_cols = list(args.sel_cols)
        self.do_z = args.do_z
        self.setup_paths()

    def setup_paths(self):
        proj_root = Path(__file__).resolve().parents[1]
        tbl_dir = proj_root / "data" / "tabular"
        self.full_df_path     = tbl_dir / "df_merged.csv"
        self.preproc_df_path  = tbl_dir / "df_preproc.csv"
        self.scores_df_path   = tbl_dir / f"df_scores_{self.score_name}.csv"
        self.scores_json_path = tbl_dir / "cog_scores.json"


def parse_args(argv: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the scores the age predictions are evaluated against, "
                    "and average a few of them into the cognitive score the models also predict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--sel_cols", nargs="+", default=COG_SCORE_COLS, metavar="COLS",
                        help="Names of columns used to calculate the cognitive score; " + 
                             "will be excluded for the PAD correlations")
    
    parser.add_argument("--no_z", dest="do_z", action="store_false", 
                        help="Average the columns as they stand, instead of standardizing each of them first")

    return parser.parse_args(argv)


def load_scores(config: Config) -> pd.DataFrame:
    '''
    Load every column of the table whose name contains the string `config.score_key`
    along with the participants' age and ID.
    '''
    cols = pd.read_csv(config.full_df_path, nrows=0).columns
    sel_cols = [ c for c in cols if config.score_key in c ]
    assert sel_cols, f"\nNo column of {config.full_df_path.name} holds '{config.score_key}'\n"
    print(f"\n{len(sel_cols)} column(s) contains the string '{config.score_key}'")

    print(f"Loading matched columns(s) from: {config.full_df_path}")
    df = pd.read_csv(config.full_df_path, index_col="BASIC_INFO_ID", usecols=["BASIC_INFO_ID", "BASIC_INFO_AGE"] + sel_cols)
    df = df.rename(columns={"BASIC_INFO_AGE": "Age"})

    all_subjs = pd.read_csv(config.preproc_df_path, usecols=["BASIC_INFO_ID"])["BASIC_INFO_ID"].tolist()
    out_df = df.reindex(all_subjs).loc[:, ["Age"] + sel_cols]
    out_df.index.name = "SID"    
    
    mask = out_df.loc[:, sel_cols].notna().any(axis=1)
    print_missing(all_subjs, out_df.index[mask].to_list())

    return out_df


def calc_cog_score(score_df: pd.DataFrame, sel_cols: list[str], do_z: bool) -> pd.Series:
    '''
    Returns the average values of the selected score columns 
    as an index of the participants' cognitive ability/maintainence.
    A participant who misses some of the columns is scored on the ones they do have.

    If `do_z`, the selected score columns is z-scored over the participants 
    before calculating the average.
    '''
    assert len(sel_cols) > 1, f"\nthe cognitive score should be averaged from multiple columns, got {len(sel_cols)}\n"
    assert set(sel_cols).issubset(set(score_df.columns)), f"\ncolumn(s) {', '.join(set(sel_cols) - set(score_df.columns))} is not in `score_df`\n"
    
    illegal = [ c for c in sel_cols if "sum" in c.lower() ]
    assert not illegal, f"\ncolumn(s) {', '.join(illegal)} contains 'Sum', which the cognitive score should not be built on\n"

    dat = score_df.loc[:, sel_cols]
    dat = (dat - dat.mean()) / dat.std() if do_z else dat
    
    return dat.mean(axis=1)


def main(config: Config):
    score_df = load_scores(config)
    score_df.to_csv(config.scores_df_path)
    print(f"\nSaved: {config.scores_df_path}")

    cog_score = calc_cog_score(score_df, config.sel_cols, config.do_z)
    summ_out = {
        "source"    : str(config.full_df_path),
        "table"     : str(config.scores_df_path),
        "score_name": config.score_name,
        "score_key" : config.score_key,
        "columns"   : config.sel_cols,
        "zscored"   : config.do_z,
        "n_subjs"   : int(cog_score.notna().sum()),
        "r_with_age": float(cog_score.corr(score_df["Age"])),
        "scores"    : cog_score
    }
    with open(config.scores_json_path, "w") as f:
        json.dump(to_json_compatible(summ_out), f, indent=2, allow_nan=False)
    print(f"Saved: {config.scores_json_path}\n")


if __name__ == "__main__":
    main(Config(parse_args()))