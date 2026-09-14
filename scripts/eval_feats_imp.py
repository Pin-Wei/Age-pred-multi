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

from helpers import train_eval_model
from predict_ages import Config as OrigConfig
from predict_ages import parse_args as orig_parse_args
from predict_ages import SID, SET, TARGETS, load_targets
from utils import mute_print, to_json_compatible, tee_output


FEAT_SRCS = ["targ-preds", "cross-decomp"]
MODES = [["only_one", "drop_one"], 
         ["only_one", "drop_one", "forward", "backward"]][0]

METRICS = [  # averaged over the targets; what the ranking and the organized table use
    f"{split}_{stat}"
    for split in ["Val", "Test"] for stat in ["MAE", "R2"]
]
ORIG_METRICS = [  # the original metrices as in the performance table
    f"{m}_{target}"
    for m in METRICS for target in TARGETS
]
METRIC_IDX = {  # split -> prefix of row indices in the performance table to be recorded 
    "Val" : "Val_pooled", 
    "Test": "Test"
} 


class Config(OrigConfig):
    '''
    Inherits the first-level model's hyper-parameters and path scheme.
    The ablation results land beside the one second-level model's outputs.
    '''
    def __init__(self, args: argparse.Namespace = None):
        args = parse_args([]) if args is None else args
        self.feat_src = args.feat_src
        self.lv2_key = args.lv2_key
        super().__init__(args)

        self.modes = list(args.modes)
        self.metrics = METRICS
        self.score_by = args.score_by
        self.score_sign = {"MAE": 1, "R2": -1}[self.score_by.split("_")[1]]

    def setup_model_params(self, args):
        super().setup_model_params(args)  # --seed, --model-lv2, --n-jobs, --verbose, ... are handled there
        if args.seeds:
            self.seeds = list(args.seeds)
        else:
            rng = np.random.default_rng()
            arr = np.arange(0, 10000)
            arr = arr[arr != self.seed]
            self.seeds = [self.seed] + rng.choice(arr, size=args.n_seeds - 1, replace=False).tolist()

        self.impute_data = args.impute_data

    def setup_vars_and_paths(self, args):
        super().setup_vars_and_paths(args)
        self.lv2_key = self.lv2_key or self._latest_lv2_key()
        self.lv2_res_dir = self.lv1_res_dir / self.lv2_key

        self.feat_tbl_path = {
            "targ-preds"  : self.lv2_res_dir / self.pred_out_path.name, 
            "cross-decomp": self.tbl_dir / "df_pls-feats.csv"
        }[self.feat_src]

        self.block_pattern = {
            "targ-preds"  : rf"^(?:{'|'.join(TARGETS)})_(?!Final$)(?P<block>.+)$",  # 1 column per target per block
            "cross-decomp": r"^(?P<block>.+)_PLS\d+$"  # k columns per block
        }[self.feat_src]

        out_dir = self.lv2_res_dir / f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.perf_main_path = out_dir / "performance_organized.csv"
        self.perf_long_path = out_dir / "performance_listed.csv"
        self.perf_path      = out_dir / "_perf_temp.csv"  # will be deleted
        self.pred_path_tmpl = out_dir / "predictions_seed-{}.csv"
        self.summ_path      = out_dir / "summary.json"
        self.log_path       = out_dir / "logs.txt"

    def _latest_lv2_key(self):
        '''
        Name of the most recent second-level run folder holding predictions of this model / seed
        '''
        found = sorted( p.parent.name for p in self.lv1_res_dir.glob(f"{self.model_lv2}_{self.seed}_*/{self.pred_out_path.name}") )
        assert found, f"\nNo '{self.pred_out_path.name}' of a {self.model_lv2} / seed {self.seed} run under {self.lv1_res_dir}"
        return found[-1]


class Lv2Data:
    '''
    Packaged data for second-level model(s)
    '''
    def __init__(self, df: pd.DataFrame, blocks: dict[str, list[str]]):
        self.df = df  # first-level models' output table; holds SID, SET, "Age" plus several feature columns
        self.blocks = blocks  # {block_name: cols_match_block_pattern}; in table order
        self.block_names = list(blocks.keys())

        self.y = df.loc[:, TARGETS].astype(np.float32)
        sets = df[SET].to_numpy(dtype=str)
        self.idx_tr = np.where(sets == "train")[0]
        self.idx_te = np.where(sets == "test")[0]

        stds = self.y.iloc[self.idx_tr].std()  # as in TargetScaler
        self.scales = (stds / stds.iloc[0]).to_dict()

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

    def id_table(self) -> pd.DataFrame:
        '''
        Participant identifiers and the targets, to be carried by every prediction table
        '''
        cols = [ c for c in [SID, SET] + TARGETS if c in self.df.columns ]
        return self.df[cols].copy()


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
    grp_data.add_argument("--feat_src", choices=FEAT_SRCS, default=FEAT_SRCS[0],
                          help="which feature table feeds the second-level model")
    grp_data.add_argument("--lv2_key", default=None,
                          help="second-level run folder under the first-level results dir; None picks the latest matching run")

    grp_search = parser.add_argument_group("search")
    grp_search.add_argument("--modes", nargs="+", choices=MODES, default=MODES, metavar="MODE",
                            help=f"searches to run, in order; from {MODES}")
    grp_search.add_argument("--score_by", choices=METRICS + ORIG_METRICS, default=f"Test_R2",
                            help="metric that drives the stepwise choices and the importance deltas")

    grp_refit = parser.add_argument_group("refitting")
    grp_refit.add_argument("--seeds", nargs="+", type=int, default=None, metavar="SEED",
                           help="explicit seeds to refit each subset with; None draws --seed plus random ones")
    grp_refit.add_argument("--n_seeds", type=int, default=5,
                           help="number of seeds to draw when --seeds is not given")
    grp_refit.add_argument("--no_impute", dest="impute_data", action="store_false",
                           help="do not impute missing feature values before fitting")

    quieter = {"verbose": 0}  # every subset is refit once per seed; keep the fits quiet unless asked
    return orig_parse_args(argv, parser, quieter)


def load_lv2_data(config: Config) -> Lv2Data:
    '''
    Read the feature table, group its columns into blocks by `config.block_pattern`,
    and bring in "Age" / SET from the preprocessed table when the features lack them.
    '''
    df = pd.read_csv(config.feat_tbl_path)
    print(f"\nFrom: {config.feat_tbl_path}")

    blocks = {}
    for col in df.columns:
        matched = re.match(config.block_pattern, col)
        if matched:
            blocks.setdefault(matched.group("block"), []).append(col)

    assert blocks, f"\nNo column of {config.feat_tbl_path.name} matches r'{config.block_pattern}'\n"

    if not {SET, *TARGETS}.issubset(df.columns):
        subj_df = load_targets(config).reset_index()
        df = pd.merge(subj_df, df, on=SID, how="inner")

    data = Lv2Data(df, blocks)
    print(f"{len(df)} participants ({len(data.idx_tr)} train, {len(data.idx_te)} test)")
    print(f"{len(blocks)} feature blocks, {sum(map(len, blocks.values()))} features:")
    for name, cols in blocks.items():
        print(f"\t{name:<14s} {len(cols):>4d} column(s), {int(df[cols].isna().any(axis=1).sum()):>4d} participant(s) missing")
    
    return data


def evaluate_subset(blocks_selected: list[str], data: Lv2Data, config: Config, cache: dict) -> dict:
    '''
    Fit a second-level model on the columns of `blocks_selected` once per seed, 
    and save their predictions and performance scores into a record.

    The record is also saved in `cache` with `frozenset(blocks_selected)` as key
    If the key is found in cache, load the record back instead of re-produce it.
    '''
    blocks_selected = data.orig_order(blocks_selected)
    key = frozenset(blocks_selected)  # an immutable and hashable version of a standard set

    if key in cache:
        return cache[key]

    cols, X = data.get_subset(blocks_selected)
    runs, perfs, preds, preds_ac = [], {}, {}, {}

    for seed in config.seeds:  # evaluate with different seeds
        with mute_print(2):
            pred_y, y_pred_ac, _ = train_eval_model(
                X, data.y, data.idx_tr, data.idx_te,
                model_type=config.model_lv2,
                seed=seed,
                seed_inner=config.seed_inner,
                n_folds=config.n_folds,
                l1_ratios=config.l1_ratios,
                alphas=config.alphas,
                max_iter=config.max_iter, 
                opt_trials=config.opt_trials, 
                n_jobs=config.n_jobs,
                verbose=config.verbose,
                impute_data=config.impute_data,
                apply_correction=True,
                model_perf_path=config.perf_path
            )

        perfs[seed] = pd.read_csv(config.perf_path, index_col="Split")
        preds[seed] = pred_y
        preds_ac[seed] = y_pred_ac
        orig_scores = {  # e.g., "Val_MAE_Age": perfs[seed].loc["Val_pooled", "MAE_Age"] 
            m: perfs[seed].loc[METRIC_IDX[m.split("_")[0]], m.split("_", 1)[1]] 
            for m in ORIG_METRICS
        }
        fair_scores = {  # m.split("_")[-1] is target
            m: orig_scores[m] / (data.scales[m.split("_")[-1]] if "MAE" in m else 1.)
            for m in ORIG_METRICS
        }
        runs.append({
            "Seed": seed, 
            **{ m: float(np.mean([ fair_scores[f"{m}_{t}"] for t in TARGETS ])) for m in METRICS }, 
            **orig_scores
        })

    runs_df = pd.DataFrame(runs)
    record = {
        "Blocks"     : blocks_selected,
        "Columns"    : cols,
        "N_blocks"   : len(blocks_selected),
        "N_feats"    : len(cols),
        "Runs"       : runs,  # [{seed + key information in performance table}, ...]
        "Performance": { s: p.to_dict(orient="index") for s, p in perfs.items() },  # {seed: {full performance table to dict}, ...}
        "Preds"      : preds,  # {seed: predicted ages aligned with data.df's rows}; dropped before the summary is dumped
        "Preds_AC"   : preds_ac,  # the same, age-corrected; dropped alongside "Preds"
        **{ k: float(runs_df[k].mean()) for k in config.metrics },  # average across runs
        **{ f"{k}_SD": float(runs_df[k].std(ddof=1)) if len(runs) > 1 else np.nan for k in config.metrics }  # SD among runs
    }
    cache[key] = record

    print(f"{len(cols)} feature(s); " + ", ".join([ f"{k} = {record[k]:.3f}" for k in config.metrics ]))

    return record


def run_ablation(mode: str, data: Lv2Data, evaluate) -> list[Entry]:
    '''
    For mode "drop_one", score the subset without the block over all `data.block_names`
    For mode "only_one", score the subset with only the block ...
    '''
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
    '''
    For mode "forward", greedily add the block that helps most, one per step.
    For mode "backward", greedily drop the block that hurts least, one per step.
    '''
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


def make_organized_table(entries: list[Entry], full_rec: dict, config: Config) -> pd.DataFrame:
    '''
    Prepend `full_rec` as a "full" `Entry` in `entries`, 
    then flatten each entry's record into one row: 
    its identity, subset sizes, metrics with SDs, and score delta against `full_rec`. 
    Rank the rows within each mode, and sort by (mode, rank).
    '''
    entries = [ Entry("full", "(all)", None, full_rec), *entries ] 
    delta_col = f"d{config.score_by}"
    out_dicts = []

    for entry in entries:
        r = entry.record
        out_dicts.append({
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

    out_df = pd.DataFrame(out_dicts)

    out_df["Rank"] = out_df["Step"]  # stepwise modes are ranked by construction; forward picks the best first
    
    rids_bw = (out_df["Mode"] == "backward")  # row indices for backward mode
    if rids_bw.any():
        steps = out_df.loc[rids_bw, "Step"]
        out_df.loc[rids_bw, "Rank"] = steps.max() + 1 - steps  # backward drops the least useful first, so reverse it

    for mode, ascending in [("drop_one", False), ("only_one", True)]:
        rids = (out_df["Mode"] == mode)
        out_df.loc[rids, "Rank"] = out_df.loc[rids, delta_col].rank(ascending=ascending, method="first")  # rank ablation modes by loss

    out_df[["Step", "Rank"]] = out_df[["Step", "Rank"]].astype("Int64")

    mode_levels = ["full", *config.modes]  # "full" first, then the modes as configured
    out_df["Mode"] = pd.Categorical(out_df["Mode"], categories=mode_levels, ordered=True)

    return out_df.sort_values(["Mode", "Rank"], kind="stable").reset_index(drop=True)


def make_pred_tables(entries: list[Entry], full_rec: dict, data: Lv2Data, config: Config) -> dict[int, pd.DataFrame]:
    '''
    For each refitting seed,
    initialize a table with participants' identifiers (SID, "Age", and SET),
    loop over `entries` (with `full_rec` prepended)
    and save the model's raw predictions (e.g., "Age_*") into a column
    and age-bias corrected ones into another (e.g., "C-Age_*")
    each named after the target, the mode and evaluated subset (see `_name_subset`).
    '''
    def _name_subset(entry: Entry) -> str:
        if entry.mode == "full":
            return "full"
        if entry.step is None:  # the ablation modes
            mode_name = {"only_one": "only", "drop_one": "drop"}[entry.mode]
            return f"{mode_name}_{entry.block}"
        return f"{entry.mode}_step-{entry.step:02d}_{entry.block}"  # the stepwise modes may revisit a block

    entries = [ Entry("full", "(all)", None, full_rec), *entries ]
    out_dict = {}

    for seed in config.seeds:
        df = data.id_table()

        for entry in entries:
            subset = _name_subset(entry)
            for target in TARGETS:
                df[f"{target}_{subset}"] = entry.record["Preds"][seed][target].to_numpy()
                df[f"C-{target}_{subset}"] = entry.record["Preds_AC"][seed][target].to_numpy()

        out_dict[seed] = df

    return out_dict


def main(config: Config):
    config.summ_path.parent.mkdir(parents=True, exist_ok=True)  # every output of this run lands in that one folder
    
    ## Load data as Lv2Data 
    data = load_lv2_data(config)

    ## Initialize `cache` and callable function `evaluate`
    cache = {}
    evaluate = lambda sel: evaluate_subset(sel, data, config, cache)  # Callable[[list[str]], dict]
    
    print(f"\nFit the reference {config.model_lv2} model on all {len(data.block_names)} blocks ...")
    full_rec = evaluate(data.block_names)

    entries = []
    for mode in config.modes:
        if mode in ["drop_one", "only_one"]:
            entries += run_ablation(mode, data, evaluate)
        else:
            entries += run_stepwise(mode, data, evaluate, config)

    config.perf_path.unlink(missing_ok=True)  # delete file

    ## Organize the full/ablation/stepwise evaluation records into ranked table and save it
    df_organized = make_organized_table(entries, full_rec, config)
    df_organized.to_csv(config.perf_main_path, index=False)
    print(f"\nSaved: {config.perf_main_path}\n")

    ## Save the records that are not collapsed over seeds
    df_listed = pd.DataFrame([ {
            "Blocks": " + ".join(record["Blocks"]), 
            "N_blocks": record["N_blocks"], 
            "N_feats": record["N_feats"], 
            **runs
        } for record in cache.values() for runs in record["Runs"]
    ])
    df_listed.to_csv(config.perf_long_path, index=False)
    print(f"Saved: {config.perf_long_path}\n")

    ## Save predictions 
    pred_dfs = make_pred_tables(entries, full_rec, data, config)
    n_id_cols = len(data.id_table().columns)
    for seed, pred_df in pred_dfs.items():
        pred_path = Path(str(config.pred_path_tmpl).format(seed))
        pred_df.to_csv(pred_path, index=False)
        n_models = (pred_df.shape[1] - n_id_cols) // (2 * len(TARGETS))
        print(f"Saved: {pred_path}")
        print(f"({n_models} model(s), {len(TARGETS)} target(s), raw and corrected)\n")

    ## Save the run's settings, the data it saw, and every evaluated record into one summary
    keep = lambda record: { k: v for k, v in record.items() if k not in ["Preds", "Preds_AC"] }  # kept in the tables above, not here
    summ_out = {
        **{ k: getattr(config, k) for k in [
            "feat_src", "lv2_key", "feat_tbl_path", "block_pattern", "modes", "score_by", 
            "model_lv1", "model_lv2", "n_folds", "seeds", "seed_inner", "l1_ratios", "alphas", "max_iter", "impute_data"
        ] },
        "n_participants": len(data.df),
        "n_train"       : len(data.idx_tr),
        "n_test"        : len(data.idx_te),
        "n_fits"        : len(df_listed),  # one fit per (distinct subset, seed) pair
        "blocks"        : data.blocks,
        "full"          : keep(full_rec),
        "importance"    : df_organized.to_dict(orient="records"),
        "subsets"       : [ keep(record) for record in cache.values() ]
    }
    with open(config.summ_path, "w") as f:
        json.dump(to_json_compatible(summ_out), f, allow_nan=False)
    print(f"Saved: {config.summ_path}\n")

    ## Print the ranked table, one section per mode
    shown = ["Rank", "Block", "N_blocks", "N_feats", *config.metrics, f"d{config.score_by}"]
    for mode in df_organized["Mode"].unique():
        print(f"\n{mode}\n{df_organized[df_organized['Mode'] == mode][shown].to_string(index=False, float_format='%.3f')}")
    print()


if __name__ == "__main__":
    config = Config(parse_args())
    log_path = Path(config.log_path)

    with tee_output(log_path):
        print(f"\nStart at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        t_0 = time.perf_counter()
        main(config)
        elapsed = time.perf_counter() - t_0
        print(f"\nFinish at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, elapsed time: {elapsed:.1f} sec")

    print(f"\nDone! logs is saved to: {log_path}")

