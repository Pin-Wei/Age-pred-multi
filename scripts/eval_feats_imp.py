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

from helpers import get_latest_results
from predict_ages import Config as OrigConfig
from predict_ages import parse_args as orig_parse_args
from predict_ages import SID, SET, AGE, COG, TARGETS, load_targets
from training import XGB_PARAM_SPACE, train_eval_model
from utils import mute_print, to_json_compatible, tee_output


FEAT_SRCS = ["targ-preds", "cross-decomp"]
ALL_MODES = ["only_one", "drop_one", "forward", "backward"]
MODES = [ALL_MODES[0], ALL_MODES[1]]

METRICS = [  # averaged over the targets; what the ranking and the organized table use
    f"{split}_{stat}"
    for split in ["Val", "Test"] for stat in ["MAE", "R2"]
]
SPLIT_IDX_MAP = {  # {split: row index in the performance table}
    "Val" : "Val_pooled",
    "Test": "Test",
}
TARGET_PREFIX_MAP = {  # {prefix of the first-level folder: targets}
    "1y": TARGETS[0],
    "2y": TARGETS[1],
}


class Config(OrigConfig):
    '''
    Inherits the first-level model's hyperparameters and path scheme.
    The ablation results land beside the one second-level model's outputs.
    '''
    def __init__(self, args: argparse.Namespace | None = None):
        args = parse_args([]) if args is None else args
        super().__init__(args)

        if self.model_lv2 == "xgboost":
            self.xgb_params = self._lv2_xgb_params()

        self.modes = list(args.modes)
        self.metrics = METRICS
        self.orig_metrics = [  # the original metrics as in the performance table
            f"{m}_{target}"
            for m in METRICS for target in self.targets
        ]
        self.score_by = args.score_by
        self.score_sign = {"MAE": 1, "R2": -1}[self.score_by.split("_")[1]]

    def setup_model_params(self, args: argparse.Namespace) -> None:
        super().setup_model_params(args)

        if args.seeds:
            self.seeds = list(args.seeds)
        else:
            rng = np.random.default_rng(self.seed)  # deterministic for a given --seed
            pool = np.arange(0, 10000)
            pool = pool[pool != self.seed]
            extra = rng.choice(pool, size=max(0, args.n_seeds - 1), replace=False).tolist()
            self.seeds = [self.seed, *extra]

        self.impute_data = args.impute_data

    def setup_vars_and_paths(self, args: argparse.Namespace) -> None:
        super().setup_vars_and_paths(args)

        self.lv2_res_dir = (
            Path(args.lv2_dir) if args.lv2_dir is not None else
            get_latest_results(search_level="lv2", lv2_key=args.lv2_key or "*")
        )
        self.lv2_key = self.lv2_res_dir.name
        self.model_lv2 = self.lv2_key.split("_")[0]

        lv1_res_dir = self.lv2_res_dir.parent
        lv1_key = lv1_res_dir.name
        lv1_parts = lv1_key.split("_")
        target_prefix = lv1_parts[0]
        self.model_lv1 = lv1_parts[1]
        self.lv2_mdl_dir = self.proj_root / "models" / lv1_key / self.lv2_key

        # Infer targets from lv1_key:
        if target_prefix in TARGET_PREFIX_MAP:
            inferred_targets = TARGET_PREFIX_MAP[target_prefix]
            if self.targets != inferred_targets:
                print(
                    f"\nInferred targets={inferred_targets} "
                    f"from folder '{lv1_key}' (overriding CLI --targets)."
                )
                self.targets = inferred_targets
                self.target_key = target_prefix

        self.feat_src = args.feat_src
        self.feat_tbl_path = {
            "targ-preds"  : self.lv2_res_dir / self.pred_out_path.name,
            "cross-decomp": self.tbl_dir / "df_pls-feats.csv",
        }[self.feat_src]

        targ_alt = "|".join(re.escape(t) for t in self.targets)
        self.block_pattern = {
            "targ-preds"  : rf"^(?:{targ_alt})_(?!Final$)(?P<block>.+)$",  # 1 column per target per block
            "cross-decomp": r"^(?P<block>.+)_PLS\d+$",  # k columns per block
        }[self.feat_src]

        out_dir = self.lv2_res_dir / f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.perf_main_path = out_dir / "performance_organized.csv"
        self.perf_long_path = out_dir / "performance_listed.csv"
        self.perf_path      = out_dir / "_perf_temp.csv"  # will be deleted
        self.pred_path_tmpl = out_dir / "predictions_seed-{}.csv"
        self.summ_path      = out_dir / "summary.json"
        self.log_path       = out_dir / "logs.txt"

    def _lv2_xgb_params(self) -> dict:
        model_paths = sorted(self.lv2_mdl_dir.glob("pipeline_*.joblib"))
        if not model_paths:
            return self.xgb_params

        import joblib
        keys = set(XGB_PARAM_SPACE.keys()) | set(self.xgb_params.keys())
        xgb_params = None

        for path in model_paths:
            model = joblib.load(path).named_steps["model"]
            model = getattr(model, "regressor_", model)
            all_params = model.get_params()
            params = { k: all_params[k] for k in keys if k in all_params }
            if xgb_params is not None and params != xgb_params:
                print("\nMismatched XGBoost hyperparameters across folds; refitting instead of reusing.")
                return self.xgb_params
            xgb_params = params

        return xgb_params if xgb_params is not None else self.xgb_params


class Lv2Data:
    '''
    Packaged data for second-level model(s).
    '''
    def __init__(self, df: pd.DataFrame, targets: list[str], blocks: dict[str, list[str]]):
        self.df = df  # first-level models' output table; holds SID, SET, targets, and feature columns
        self.blocks = blocks  # {block_name: cols_match_block_pattern}; in table order
        self.block_names = list(blocks.keys())

        self.y = df.loc[:, targets].astype(np.float32)
        sets = df[SET].to_numpy(dtype=str)
        self.idx_tr = np.flatnonzero(sets == "train")
        self.idx_te = np.flatnonzero(sets == "test")

        stds = self.y.iloc[self.idx_tr].std().replace(0.0, 1.0)  # as in TargetScaler
        self.scales = (stds / stds.iloc[0]).to_dict()

    def orig_order(self, blocks_selected: list[str]) -> list[str]:
        '''
        Return names of selected blocks in df's original order.
        '''
        wanted = set(blocks_selected)

        return [ b for b in self.block_names if b in wanted ]

    def get_subset(self, blocks_selected: list[str]) -> tuple[list[str], np.ndarray]:
        '''
        Return column names and values of selected blocks, in df's original order.
        '''
        cols = [ c for b in self.orig_order(blocks_selected) for c in self.blocks[b] ]

        return cols, self.df[cols].to_numpy(dtype=np.float32)

    def id_table(self) -> pd.DataFrame:
        '''
        Participant identifiers and the targets, to be carried by every prediction table.
        '''
        cols = [ c for c in [SID, SET, AGE, COG] if c in self.df.columns ]

        return self.df[cols].copy()


class Entry(NamedTuple):
    '''
    One evaluated subset, tagged with the search that asked for it.
    '''
    mode  : str
    block : str
    step  : int | None
    record: dict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    '''
    Command-line overrides for the Config values worth varying between runs.
    The inherited Config's own options (--seed, --model_lv2, --alphas, ...) are added by `predict_ages.parse_args`.
    '''
    parser = argparse.ArgumentParser(
        description="Evaluate feature-block importance for a second-level age-prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grp_data = parser.add_argument_group("data source")
    grp_data.add_argument("--feat_src", choices=FEAT_SRCS, default=FEAT_SRCS[0], 
                          help="Which type of first-level outputs should be used as the input to the second-level model")
    grp_data.add_argument("--lv2_key", default=None,
                          help="Name of a second-level run folder (glob pattern accepted). None takes the most recent run")
    grp_data.add_argument("--lv2_dir", default=None,
                          help="Path to a second-level model's result folder")

    grp_search = parser.add_argument_group("search")
    grp_search.add_argument("--modes", nargs="+", choices=ALL_MODES, default=MODES, metavar="MODE",
                            help=f"Search modes to run, in order; from {ALL_MODES}")
    grp_search.add_argument("--score_by", choices=METRICS, default="Test_R2",
                            help="Metric that drives the stepwise choices and the importance deltas")

    grp_refit = parser.add_argument_group("refitting")
    grp_refit.add_argument("--seeds", nargs="+", type=int, default=None, metavar="SEED",
                           help="Explicit seeds to refit each subset with; None draws --seed plus random ones")
    grp_refit.add_argument("--n_seeds", type=int, default=5,
                           help="Number of seeds to draw when --seeds is not given")
    grp_refit.add_argument("--no_impute", dest="impute_data", action="store_false",
                           help="Do not impute missing feature values before fitting")

    quieter = {"verbose": 0}  # every subset is refit once per seed; keep the fits quiet unless asked

    return orig_parse_args(argv, parser, quieter)


def load_lv2_data(config: Config) -> Lv2Data:
    '''
    Read the feature table, group its columns into blocks by `config.block_pattern`,
    and bring in targets / SET from the preprocessed table when the features lack them.
    '''
    if not config.feat_tbl_path.is_file():
        raise FileNotFoundError(f"Feature table not found: {config.feat_tbl_path}")

    df = pd.read_csv(config.feat_tbl_path)
    print(f"\nFrom: {config.feat_tbl_path}")

    blocks: dict[str, list[str]] = {}
    for col in df.columns:
        matched = re.match(config.block_pattern, col)
        if matched:
            blocks.setdefault(matched.group("block"), []).append(col)

    if not blocks:
        raise ValueError(f"No column of {config.feat_tbl_path.name} matches r'{config.block_pattern}'.")

    needed_meta = {SET, *config.targets}
    if not needed_meta.issubset(df.columns):
        subj_df = load_targets(config).reset_index()
        extra_cols = [ c for c in subj_df.columns if c == SID or c not in df.columns ]
        df = pd.merge(subj_df[extra_cols], df, on=SID, how="inner")

    data = Lv2Data(df, config.targets, blocks)
    print(f"{len(df)} participants ({len(data.idx_tr)} train, {len(data.idx_te)} test)")
    print(f"{len(blocks)} feature blocks, {sum(map(len, blocks.values()))} features:")
    for name, cols in blocks.items():
        n_miss = int(df[cols].isna().any(axis=1).sum())
        print(f"\t{name:<14s} {len(cols):>4d} column(s), {n_miss:>4d} participant(s) missing")

    return data


def evaluate_subset(
    blocks_selected: list[str],
    data: Lv2Data,
    config: Config,
    cache: dict,
) -> dict:
    '''
    Fit a second-level model on the columns of `blocks_selected` once per seed,
    and save their predictions and performance scores into a record.
    '''
    blocks_selected = data.orig_order(blocks_selected)
    key = frozenset(blocks_selected)

    if key in cache:
        return cache[key]

    cols, X = data.get_subset(blocks_selected)
    runs, perfs, preds, preds_ac = [], {}, {}, {}
    apply_correction = config.n_folds > 1
    base_kwargs = config.common_train_kwargs(config.model_lv2)

    for seed in config.seeds:
        with mute_print(2):
            pred_y, y_pred_ac, _ = train_eval_model(
                X, data.y, data.idx_tr, data.idx_te,
                **{**base_kwargs, "seed": seed},
                impute_data=config.impute_data,
                apply_correction=apply_correction,
                model_perf_path=config.perf_path,
            )

        perf_df = pd.read_csv(config.perf_path, index_col="Split")
        perfs[seed] = perf_df
        preds[seed] = pred_y
        preds_ac[seed] = y_pred_ac

        orig_scores = {}
        for m in config.orig_metrics:
            split, metric_col = m.split("_", 1)
            row_key = SPLIT_IDX_MAP[split]
            if row_key not in perf_df.index and split == "Val":
                row_key = "Train"  # fallback when n_folds <= 1
            orig_scores[m] = float(perf_df.loc[row_key, metric_col])

        fair_scores = {
            m: orig_scores[m] / (data.scales[m.rsplit("_", 1)[-1]] if "MAE" in m else 1.0)
            for m in config.orig_metrics
        }
        runs.append({
            "Seed": seed,
            **{ m: float(np.mean([ fair_scores[f"{m}_{t}"] for t in config.targets ])) for m in METRICS },
            **orig_scores,
        })

    runs_df = pd.DataFrame(runs)
    record = {
        "Blocks"     : blocks_selected,
        "Columns"    : cols,
        "N_blocks"   : len(blocks_selected),
        "N_feats"    : len(cols),
        "Runs"       : runs,
        "Performance": { s: p.to_dict(orient="index") for s, p in perfs.items() },
        "Preds"      : preds,     # dropped before the summary is dumped
        "Preds_AC"   : preds_ac,  # dropped alongside "Preds"
        **{ k: float(runs_df[k].mean()) for k in config.metrics },
        **{ f"{k}_SD": float(runs_df[k].std(ddof=1)) if len(runs) > 1 else np.nan for k in config.metrics },
    }
    cache[key] = record

    print(f"{len(cols)} feature(s); " + ", ".join(f"{k} = {record[k]:.3f}" for k in config.metrics))

    return record


def run_ablation(mode: str, data: Lv2Data, evaluate) -> list[Entry]:
    '''
    For mode "drop_one", score the subset without the block over all `data.block_names`.
    For mode "only_one", score the subset with only the block.
    '''
    entries = []

    for i, block in enumerate(data.block_names, start=1):
        if mode == "drop_one":
            selected_blocks = [ b for b in data.block_names if b != block ]
            prep = "without"
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
    is_forward = mode == "forward"

    def _move(chosen: list[str], block: str) -> list[str]:
        if is_forward:
            return data.orig_order(chosen + [block])
        return [ b for b in chosen if b != block ]

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
        print(
            f"\n[{mode} step {step}]: {'+' if is_forward else '-'} {best}"
            f" -> {config.score_by} = {scored[best][config.score_by]:.3f}"
            f" ({len(chosen)} block(s) kept)"
        )

    if not is_forward:  # the survivor is never dropped; record it here to keep every block ranked
        entries.append(Entry(mode, chosen[0], n_steps + 1, evaluate(chosen)))

    return entries


def make_organized_table(entries: list[Entry], full_rec: dict, config: Config) -> pd.DataFrame:
    '''
    Prepend `full_rec` as a "full" `Entry` in `entries`, flatten each entry into one row,
    compute score delta against `full_rec`, rank within each mode, and sort by (mode, rank).
    '''
    entries = [Entry("full", "(all)", None, full_rec), *entries]
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
            "Blocks"  : " + ".join(r["Blocks"]),
        })

    out_df = pd.DataFrame(out_dicts)
    out_df["Rank"] = out_df["Step"]

    rids_bw = out_df["Mode"] == "backward"
    if rids_bw.any():
        steps = out_df.loc[rids_bw, "Step"]
        out_df.loc[rids_bw, "Rank"] = steps.max() + 1 - steps

    for mode, ascending in [("drop_one", False), ("only_one", True)]:
        rids = out_df["Mode"] == mode
        out_df.loc[rids, "Rank"] = out_df.loc[rids, delta_col].rank(ascending=ascending, method="first")

    out_df[["Step", "Rank"]] = out_df[["Step", "Rank"]].astype("Int64")
    out_df["Mode"] = pd.Categorical(out_df["Mode"], categories=["full", *config.modes], ordered=True)

    return out_df.sort_values(["Mode", "Rank"], kind="stable").reset_index(drop=True)


def make_pred_tables(
    entries: list[Entry],
    full_rec: dict,
    data: Lv2Data,
    config: Config,
) -> dict[int, pd.DataFrame]:
    '''
    Build one prediction DataFrame per refitting seed holding raw and age-corrected
    predictions across all evaluated subsets.
    '''
    def _name_subset(entry: Entry) -> str:
        if entry.mode == "full":
            return "full"
        if entry.step is None:
            mode_name = {"only_one": "only", "drop_one": "drop"}[entry.mode]
            return f"{mode_name}_{entry.block}"
        return f"{entry.mode}_step-{entry.step:02d}_{entry.block}"

    entries = [Entry("full", "(all)", None, full_rec), *entries]
    id_df = data.id_table()
    out_dict = {}

    for seed in config.seeds:
        new_cols = {}
        for entry in entries:
            subset = _name_subset(entry)
            pred_df = entry.record["Preds"][seed]
            pred_ac_df = entry.record["Preds_AC"][seed]
            for target in config.targets:
                new_cols[f"{target}_{subset}"] = pred_df[target].to_numpy()
                if pred_ac_df is not None:
                    new_cols[f"C-{target}_{subset}"] = pred_ac_df[target].to_numpy()

        df = pd.concat([id_df, pd.DataFrame(new_cols, index=id_df.index)], axis=1)
        out_dict[seed] = df

    return out_dict


def main(config: Config) -> None:
    config.summ_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_lv2_data(config)
    cache: dict = {}
    evaluate = lambda sel: evaluate_subset(sel, data, config, cache)

    try:
        print(f"\nFit the reference {config.model_lv2} model on all {len(data.block_names)} blocks ...")
        full_rec = evaluate(data.block_names)

        entries = []
        for mode in config.modes:
            if mode in {"drop_one", "only_one"}:
                entries += run_ablation(mode, data, evaluate)
            else:
                entries += run_stepwise(mode, data, evaluate, config)
    finally:
        config.perf_path.unlink(missing_ok=True)

    df_organized = make_organized_table(entries, full_rec, config)
    df_organized.to_csv(config.perf_main_path, index=False)
    print(f"\nSaved: {config.perf_main_path}\n")

    df_listed = pd.DataFrame([
        {
            "Blocks"  : " + ".join(record["Blocks"]),
            "N_blocks": record["N_blocks"],
            "N_feats" : record["N_feats"],
            **run,
        }
        for record in cache.values() for run in record["Runs"]
    ])
    df_listed.to_csv(config.perf_long_path, index=False)
    print(f"Saved: {config.perf_long_path}\n")

    pred_dfs = make_pred_tables(entries, full_rec, data, config)
    n_subsets = len(entries) + 1
    for seed, pred_df in pred_dfs.items():
        pred_path = Path(str(config.pred_path_tmpl).format(seed))
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved: {pred_path}")
        print(f"({n_subsets} subset(s), {len(config.targets)} target(s))\n")

    keep = lambda record: { k: v for k, v in record.items() if k not in {"Preds", "Preds_AC"} }
    summ_out = {
        **{
            k: getattr(config, k) for k in [
                "targets", "feat_src", "lv2_key", "feat_tbl_path",
                "block_pattern", "modes", "score_by", "model_lv1", "model_lv2",
                "n_folds", "seeds", "seed_inner", "l1_ratios", "alphas", "max_iter", "impute_data",
            ]
        },
        "n_participants": len(data.df),
        "n_train"       : len(data.idx_tr),
        "n_test"        : len(data.idx_te),
        "n_fits"        : len(df_listed),
        "blocks"        : data.blocks,
        "full"          : keep(full_rec),
        "importance"    : df_organized.to_dict(orient="records"),
        "subsets"       : [ keep(record) for record in cache.values() ],
    }
    with open(config.summ_path, "w") as f:
        json.dump(to_json_compatible(summ_out), f, allow_nan=False)
    print(f"Saved: {config.summ_path}\n")

    shown = ["Rank", "Block", "N_blocks", "N_feats", *config.metrics, f"d{config.score_by}"]
    for mode in df_organized["Mode"].unique():
        sub = df_organized[df_organized["Mode"] == mode][shown]
        print(f"\n{mode}\n{sub.to_string(index=False, float_format='%.3f')}")
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
