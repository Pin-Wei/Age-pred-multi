#!/usr/bin/env python3


import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import utils; utils.MUTED = 2
from predict_ages import Config as OrigConfig
from predict_ages import parse_args as orig_parse_args
from predict_ages import tee_output 


FEAT_SRCS = ["age-preds", "cross-decomp"]
MODES     = ["only_one", "drop_one", "forward", "backward"]
METRICS   = ["Val_MAE", "Val_R2", "Test_MAE", "Test_R2"]


class Config(OrigConfig):
    '''
    Inherits the first-level model's hyper-parameters and path scheme.
    The ablation results land beside the one second-level model's outputs.
    '''
    def __init__(self, args: argparse.Namespace = None):
        self.args = args if args is not None else parse_args([])
        self.feat_src = self.args.feat_src
        self.lv2_key = self.args.lv2_key
        super().__init__(self.args)

        self.modes = list(self.args.modes)
        self.metrics = METRICS
        self.score_by = self.args.score_by
        self.score_sign = {"MAE": 1, "R2": -1}[self.score_by.split("_")[1]]

    def setup_model_params(self):
        super().setup_model_params()  # --seed, --model-lv2, --n-jobs, --verbose, ... are handled there
        if self.args.seeds:
            self.seeds = list(self.args.seeds)
        else:
            self.seeds = [self.seed] + np.random.randint(0, 100, size=self.args.n_seeds - 1).tolist()

        self.impute_data = self.args.impute_data

    def setup_vars_and_paths(self):
        super().setup_vars_and_paths()
        self.lv2_key = self.lv2_key or self._latest_lv2_key()
        self.lv2_res_dir = self.lv1_res_dir / self.lv2_key

        self.feat_tbl_path = {
            "age-preds"   : self.lv2_res_dir / self.pred_out_path.name, 
            "cross-decomp": self.tbl_dir / "df_pls-feats.csv"
        }[self.feat_src]

        self.block_pattern = {
            "age-preds"   : r"^Age_(?!Final$)(?P<block>.+)$",  # 1 column per block
            "cross-decomp": r"^(?P<block>.+)_PLS\d+$"          # k columns per block
        }[self.feat_src]

        out_dir = self.lv2_res_dir / f"feat_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.imp_out_path  = out_dir / "importance.csv"
        self.subs_out_path = out_dir / "subsets.csv"
        self.summ_out_path = out_dir / "summary.json"
        self.log_out_path  = out_dir / "logs.txt"
        self.perf_path     = out_dir / "_perf_temp.csv" 

    def _latest_lv2_key(self):
        found = sorted( p.parent.name for p in self.lv1_res_dir.glob(f"{self.model_lv2}_{self.seed}_*/{self.pred_out_path.name}") )
        assert found, f"\nNo '{self.pred_out_path.name}' of a {self.model_lv2} / seed {self.seed} run under {self.lv1_res_dir}"
        return found[-1]


class Lv2Data:
    '''
    Packaged data for second-level model(s)
    '''
    def __init__(self, df: pd.DataFrame, blocks: dict[str, list[str]]):
        self.df = df  # first-level models' output table; holds "SID", "Age", "Set" plus several feature columns
        self.blocks = blocks  # {block_name: cols_match_block_pattern}; in table order
        self.block_names = list(blocks.keys())

        self.y = df["Age"].to_numpy(dtype=np.float32)
        sets = df["Set"].to_numpy(dtype=str)
        self.idx_tr = np.where(sets == "train")[0]
        self.idx_te = np.where(sets == "test")[0]

    def orig_order(self, blocks_selected: list[str]) -> list[str]:
        '''
        Returns names of selected blocks in df's original order
        '''
        wanted = set(blocks_selected)
        return [ b for b in self.block_names if b in wanted ]

    def get_subset(self, blocks_selected: list[str]) -> tuple[list[str], np.ndarray]:
        '''
        Returns column names and values of selected blocks, in df's original order
        '''
        cols = [ c for b in self.orig_order(blocks_selected) for c in self.blocks[b] ]
        return cols, self.df[cols].to_numpy(dtype=np.float32)


class Entry(NamedTuple):
    '''
    One evaluated subset, tagged with the search that asked for it.
    '''
    mode  : str
    block : str
    step  : int | None
    record: dict


def parse_args(argv: list[str] = None) -> argparse.Namespace:
    '''
    Command-line overrides for the Config values worth varying between runs.
    The inherited Config's own options (--seed, --model-lv2, --alphas, ...) are added by `predict_ages.parse_args`.
    '''
    parser = argparse.ArgumentParser(
        description="Evaluate feature-block importance for a second-level age-prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    grp_data = parser.add_argument_group("data source")
    grp_data.add_argument("--feat-src", choices=FEAT_SRCS, default=FEAT_SRCS[0],
                          help="which feature table feeds the second-level model")
    grp_data.add_argument("--lv2-key", default=None,
                          help="second-level run folder under the first-level results dir; None picks the latest matching run")

    grp_search = parser.add_argument_group("search")
    grp_search.add_argument("--modes", nargs="+", choices=MODES, default=MODES, metavar="MODE",
                            help=f"searches to run, in order; from {MODES}")
    grp_search.add_argument("--score-by", choices=METRICS, default="Test_MAE",
                            help="metric that drives the stepwise choices and the importance deltas")

    grp_refit = parser.add_argument_group("refitting")
    grp_refit.add_argument("--seeds", nargs="+", type=int, default=None, metavar="SEED",
                           help="explicit seeds to refit each subset with; None draws --seed plus random ones")
    grp_refit.add_argument("--n-seeds", type=int, default=5,
                           help="number of seeds to draw when --seeds is not given")
    grp_refit.add_argument("--no-impute", dest="impute_data", action="store_false",
                           help="do not impute missing feature values before fitting")

    quieter = {"verbose": 0}  # every subset is refit once per seed; keep the fits quiet unless asked
    return orig_parse_args(argv, parser, quieter)


def load_lv2_data(config: Config) -> Lv2Data:
    df = pd.read_csv(config.feat_tbl_path)
    print(f"\nFrom: {config.feat_tbl_path}")

    blocks = {}
    for col in df.columns:
        matched = re.match(config.block_pattern, col)
        if matched:
            blocks.setdefault(matched.group("block"), []).append(col)

    assert blocks, f"\nNo column of {config.feat_tbl_path.name} matches r'{config.block_pattern}'\n"

    if not {"Age", "Set"}.issubset(df.columns):
        subj_df = pd.read_csv(config.df_preproc_path, usecols=["BASIC_INFO_ID", "BASIC_INFO_AGE", "Set"])
        subj_df.rename(columns={"BASIC_INFO_ID": "SID", "BASIC_INFO_AGE": "Age"}, inplace=True)
        df = pd.merge(subj_df, df, on="SID", how="inner")

    data = Lv2Data(df, blocks)
    print(f"{len(df)} participants ({len(data.idx_tr)} train, {len(data.idx_te)} test)")
    print(f"{len(blocks)} feature blocks, {sum(map(len, blocks.values()))} features:")
    for name, cols in blocks.items():
        print(f"\t{name:<14s} {len(cols):>4d} column(s), {int(df[cols].isna().any(axis=1).sum()):>4d} participant(s) missing")
    
    return data


def evaluate_subset(blocks_selected: list[str], data: Lv2Data, config: Config, cache: dict) -> dict:
    '''
    Fit a second-level model on the columns of `blocks_selected` and score it, once per seed.
    '''
    blocks_selected = data.orig_order(blocks_selected)
    key = frozenset(blocks_selected)  # an immutable and hashable version of a standard set

    if key in cache:
        return cache[key]

    cols, X = data.get_subset(blocks_selected)
    runs, perfs = [], {}

    for seed in config.seeds:
        t_0 = time.perf_counter()
        utils.train_eval_model(  # bare call, and then load performance metrics back through 'model_perf_path'
            X, data.y, data.idx_tr, data.idx_te,
            model_type=config.model_lv2,
            seed=seed,
            seed_inner=config.seed_inner,
            n_folds=config.n_folds,
            l1_ratios=config.l1_ratios,
            alphas=config.alphas,
            max_iter=config.max_iter,
            n_jobs=config.n_jobs,
            verbose=config.verbose,
            impute_data=config.impute_data,
            model_perf_path=config.perf_path
        )
        perfs[seed] = pd.read_csv(config.perf_path, index_col="Split")
        runs.append({
            "Seed"    : seed,
            "Val_MAE" : perfs[seed].loc["Val_pooled", "MAE"],
            "Val_R2"  : perfs[seed].loc["Val_pooled", "R2"],
            "Test_MAE": perfs[seed].loc["Test", "MAE"],
            "Test_R2" : perfs[seed].loc["Test", "R2"],
            "Seconds" : time.perf_counter() - t_0
        })

    runs_df = pd.DataFrame(runs)
    record = {
        "Blocks"     : blocks_selected,
        "Columns"    : cols,
        "N_blocks"   : len(blocks_selected),
        "N_feats"    : len(cols),
        "Runs"       : runs,
        "Seconds"    : float(runs_df["Seconds"].sum()),
        "Performance": { s: p.to_dict(orient="index") for s, p in perfs.items() },
        **{ k: float(runs_df[k].mean()) for k in config.metrics },
        **{ f"{k}_SD": float(runs_df[k].std(ddof=1)) if len(runs) > 1 else np.nan for k in config.metrics }
    }
    cache[key] = record

    print(", ".join([ f"{k} = {record[k]:.3f}" for k in config.metrics ]))
    print(f"{len(cols)} feature(s), elapsed time: {record['Seconds']:.1f} sec")

    return record


def run_ablation(mode: str, data: Lv2Data, evaluate) -> list[Entry]:
    entries = []

    for i, block in enumerate(data.block_names, start=1):
        if mode == "drop_one":
            selected_blocks = data.block_names.copy()
            selected_blocks.remove(block)
            prep = "without"  # preposition
        else:  # "only_one"
            selected_blocks = [block]
            prep = "only"

        print(f"\n[{mode} {i}/{len(data.block_names)}] {prep} {block}")
        entries.append(Entry(mode, block, None, evaluate(selected_blocks)))

    return entries


def run_stepwise(mode: str, data: Lv2Data, evaluate, config: Config) -> list[Entry]:
    def _move(chosen: list[str], block: str) -> list[str]:
        if mode == "forward":
            return data.orig_order(chosen + [block])
        else:  # "backward"
            selected_blocks = chosen.copy()
            selected_blocks.remove(block)
            return selected_blocks

    is_forward = (mode == "forward")
    chosen = [] if is_forward else list(data.block_names)
    n_steps = len(data.block_names) - (0 if is_forward else 1)
    entries = []

    for step in range(1, n_steps + 1):
        pool = [ b for b in data.block_names if b not in chosen ] if is_forward else list(chosen)
        scored = {}

        for block in pool:
            print(f"\n[{mode} step {step}/{n_steps}] {'+' if is_forward else '-'} {block}")
            scored[block] = evaluate(_move(chosen, block))

        best = min(pool, key=lambda b: config.score_sign * scored[b][config.score_by])
        chosen = _move(chosen, best)

        entries.append(Entry(mode, best, step, scored[best]))
        print(f"\n[{mode} step {step}]: {'+' if is_forward else '-'} {best}"
              f" -> {config.score_by} = {scored[best][config.score_by]:.3f}"
              f" ({len(chosen)} block(s) kept)")

    if not is_forward:  # the survivor is never dropped, so record it here to keep every block ranked
        entries.append(Entry(mode, chosen[0], n_steps + 1, evaluate(chosen)))  # a cache hit; same subset as the last step

    return entries


def make_importance_table(entries: list[Entry], full_rec: dict, config: Config) -> pd.DataFrame:
    entries = [ Entry("full", "(all)", None, full_rec), *entries ]  # a local list; the caller's stays untouched
    delta_col = f"d{config.score_by}"
    imp_dicts = []

    for entry in entries:
        r = entry.record
        imp_dicts.append({
            "Mode"    : entry.mode,
            "Block"   : entry.block,
            "Step"    : entry.step,
            "N_blocks": r["N_blocks"],
            "N_feats" : r["N_feats"],
            **{ k: r[k] for k in config.metrics },
            **{ f"{k}_SD": r[f"{k}_SD"] for k in config.metrics },
            delta_col : config.score_sign * (r[config.score_by] - full_rec[config.score_by]),
            "Blocks"  : " + ".join(r["Blocks"])
        })

    imp_df = pd.DataFrame(imp_dicts)

    imp_df["Rank"] = imp_df["Step"]  # stepwise modes are ranked by construction; forward picks the best first
    of_backward = (imp_df["Mode"] == "backward")  # but backward drops the least useful first, so reverse it
    if of_backward.any():
        steps = imp_df.loc[of_backward, "Step"]
        imp_df.loc[of_backward, "Rank"] = steps.max() + 1 - steps

    for mode, ascending in [("drop_one", False), ("only_one", True)]:  # rank ablation modes by loss
        of_mode = (imp_df["Mode"] == mode)
        imp_df.loc[of_mode, "Rank"] = imp_df.loc[of_mode, delta_col].rank(ascending=ascending, method="first")

    imp_df[["Step", "Rank"]] = imp_df[["Step", "Rank"]].astype("Int64")

    mode_levels = ["full", *config.modes]  # "full" first, then the modes as configured
    imp_df["Mode"] = pd.Categorical(imp_df["Mode"], categories=mode_levels, ordered=True)

    return imp_df.sort_values(["Mode", "Rank"], kind="stable").reset_index(drop=True)


def main(config: Config):
    config.imp_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = load_lv2_data(config)
    cache = {}
    evaluate = lambda sel: evaluate_subset(sel, data, config, cache)  # Callable[[list[str]], dict]
    
    t_0 = time.perf_counter()

    print(f"\nFit the reference {config.model_lv2} model on all {len(data.block_names)} blocks ...")
    full_rec = evaluate(data.block_names)

    entries = []
    for mode in config.modes:
        if mode in ["drop_one", "only_one"]:
            entries += run_ablation(mode, data, evaluate)
        else:
            entries += run_stepwise(mode, data, evaluate, config)

    config.perf_path.unlink(missing_ok=True)  # delete file

    imp_df = make_importance_table(entries, full_rec, config)
    imp_df.to_csv(config.imp_out_path, index=False)
    print(f"\nSaved: {config.imp_out_path}\n")

    subs_df = pd.DataFrame([ {
            "Blocks": " + ".join(record["Blocks"]), 
            "N_blocks": record["N_blocks"], 
            "N_feats": record["N_feats"], 
            **runs
        } for record in cache.values() for runs in record["Runs"]
    ])
    subs_df.to_csv(config.subs_out_path, index=False)
    print(f"Saved: {config.subs_out_path}\n")

    summ_out = {
        **{ k: getattr(config, k) for k in [
            "feat_src", "feat_tbl_path", "lv2_key", "block_pattern", "modes", "score_by", "model_lv1", "model_lv2",
            "n_folds", "seeds", "seed_inner", "l1_ratios", "alphas", "max_iter", "impute_data"
        ] },
        "n_participants": len(data.df),
        "n_train"       : len(data.idx_tr),
        "n_test"        : len(data.idx_te),
        "n_fits"        : len(subs_df),
        "seconds"       : time.perf_counter() - t_0,
        "blocks"        : data.blocks,
        "full"          : full_rec,
        "importance"    : imp_df.to_dict(orient="records"),
        "subsets"       : list(cache.values())
    }
    with open(config.summ_out_path, "w") as f:
        json.dump(utils.to_json_compatible(summ_out), f, allow_nan=False)
    print(f"Saved: {config.summ_out_path}\n")

    shown = ["Rank", "Block", "N_blocks", "N_feats", *config.metrics, f"d{config.score_by}"]
    for mode in imp_df["Mode"].unique():
        print(f"\n{mode}\n{imp_df[imp_df['Mode'] == mode][shown].to_string(index=False, float_format='%.3f')}")
    print()


if __name__ == "__main__":
    config = Config(parse_args())
    log_path = Path(config.log_out_path)

    with tee_output(log_path):
        print(f"\nStart at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main(config)
        print(f"\nFinish at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nDone! logs is saved to: {log_path}")