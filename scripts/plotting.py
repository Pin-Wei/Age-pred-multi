#!/usr/bin/env python3


import re
from contextlib import contextmanager
from math import radians, sin
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from matplotlib.transforms import Bbox


MODEL_MODES = {  # prefix of the model name -> (label in the legend, color)
    "full": ("full (U)"      , "#4C4C4C"),
    "only": ("only ({X})"    , "#4C72B0"),
    "drop": ("drop (U - {X})", "#C44E52")
}
FEAT_LABELS = {  # feature set as the model columns spell it -> what the figures print
    "BEH_COGNITIVE": "behav cog",
    "FUN_COGNITIVE": "func cog",
    "DTI_FA"       : "FA",
    "DTI_MD"       : "MD",
    "rs-MRI_FC"    : "FC",
    "rs-EEG_PSD"   : "PSD",
    "MRI_ROI"      : "ROI"
}
COEF_BAR_COLORS = {  # sign of the weight -> color
    "+": "#4C72B0",
    "-": "#C44E52"
}
SUBJ_SET_MARKS = {  # participant set -> (color, marker)
    "train": ("#4C72B0", "o"),
    "test" : ("#C44E52", "^")
}
NS_LABEL = "n.s."
BASE_FS = 16  # the base multiplicand of all variables related to font size
FS_RATIOS = {  # multipliers of rcParams related to font size
    "font.size"            : 1.0,
    "axes.titlesize"       : 1.1,  # default is "large" (1.2)
    "axes.labelsize"       : 1.1,  # default is "medium" (1.0)
    "xtick.labelsize"      : 1.0,  # default is "medium" (1.0)
    "ytick.labelsize"      : 1.0,  # default is "medium" (1.0)
    "legend.fontsize"      : 0.8,  # default is "medium" (1.0)
    "legend.title_fontsize": 0.8   # default is None (1.0)
}
CHAR_ASPECT = 0.6  # aspect ratio of a character
PAD_INCHES = 0.12  # the amount of white space padding around a figure when saving it with a cropped bounding box
FIG_BASES = {  # multiplicand of rcParams related to line width
    "lines.linewidth"  : 1.0,  # in points
    "lines.markersize" : 7.0,  # in points; marker diameter in scatter plots (both `ax.scatter` and `sns.stripplot`)
    "axes.linewidth"   : 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0, 
    "errorbar.capsize" : 3.0
}

## ========================================================================================

class Style:
    def __init__(
        self,
        fig_scale: float = 1.,
        font_scale: float = 1.,
        dpi: int = 200
    ):
        self.dpi = dpi
        self.fig_scale = fig_scale
        self.base_fs = BASE_FS * font_scale
        self.base_lw = FIG_BASES["lines.linewidth"] * fig_scale

    @property
    def rc_params(self) -> dict:
        '''
        For modifing Matplotlib's runtime configuration parameters (`matplotlib.rcParams`)
        through `matplotlib.pyplot.rc_context`. 

        For a full list of config parameters, see:
        - https://matplotlib.org/stable/users/explain/configuration.html      
        
        For usage information, see:
        - https://matplotlib.org/stable/users/explain/customizing.html#runtime-rc-settings
        '''
        rc = { 
            "axes.spines.top": False, 
            "axes.spines.right": False, 
            "legend.markerscale": 2.2, 
            "figure.dpi": self.dpi, 
            # "savefig.dpi": "figure"
        }
        rc.update({
            k: self.base_fs * ratio for k, ratio in FS_RATIOS.items()
        })
        rc.update({
            k: base * self.fig_scale for k, base in FIG_BASES.items()
        })
        return rc

    def fig_size(
        self, cell: tuple[float, float], margin: tuple[float, float], 
        n_cols: int = 0, n_rows: int = 0,
        ytick_len: float = 0., xtick_len: float = 0., ytick_fs: float = 0., xtick_fs: float = 0.
    ) -> tuple[float, float]:
        '''
        Set width and height of one figure (in inches) based on the given variables, 
        which includes:
        - `cell`: the inches one column/row of figure costs.
        - `margin`: the inches of the white space around the figure.
        - `xtick_len`/`ytick_len`: the character numbers of the longest x- or y-tick label.
        - `xtick_fs: the inches of the longest x-tick label (overrides the default one).
        '''
        (cell_w, cell_h), (margin_w, margin_h) = cell, margin
        ytick_fs = ytick_fs or self.base_fs * FS_RATIOS["ytick.labelsize"]
        xtick_fs = xtick_fs or self.base_fs * FS_RATIOS["xtick.labelsize"]
        return (
            (n_cols * cell_w + margin_w) * self.fig_scale + n_chars_2_inches(ytick_len, ytick_fs),
            (n_rows * cell_h + margin_h) * self.fig_scale + n_chars_2_inches(xtick_len, xtick_fs)
        )


def sig_stars(q: float, fdr_alpha: float) -> str:
    if not np.isfinite(q) or q >= fdr_alpha:  # not significant
        return NS_LABEL
        
    return f"{'*' * sum( q <= t for t in [.05, .01, .001] )}"


def model_name_2_label(name: str) -> str:
    if name == "full":
        return MODEL_MODES[name][0]

    if name in FEAT_LABELS:
        return FEAT_LABELS[name]

    mode, _, feat = name.partition("_")
    feat = FEAT_LABELS.get(feat, feat)

    if mode in ["only", "drop"]:
        return MODEL_MODES[mode][0].replace("X", feat).replace(f"{mode} (", "").replace(")", "")

    return name


def capitalize_upper_label(label: str) -> str:
    '''
    Convert all-capitals label to title case (e.g., "MEMORY" -> "Memory")
    otherwise left as it is (e.g., "LogMemI")
    '''
    return label.capitalize() if label.isupper() else label


def fprint_prob(prob: float) -> str:
    '''
    Write a probability without its trailing and leading zero(s) (e.g., 0.50 -> ".05")
    '''
    return f"{prob:g}".lstrip("0")


def points_2_inches(pt: float) -> float:
    return pt / 72  # 1 inch = 72 points


def n_chars_2_inches(n_chars: float, fontsize: float) -> float:
    return points_2_inches(n_chars * CHAR_ASPECT * fontsize)


@contextmanager
def create_fig(style: Style, out_path: Path, cell: tuple, margin: tuple, **size_kwargs):
    '''
    Create a figure of `fig_size` inches holding one axes, 
    with its rc parameter set using `style`,
    then lay it out, save it and close it. 
    '''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = style.fig_size(cell, margin, **size_kwargs)

    with plt.rc_context(style.rc_params):
        fig = plt.figure(figsize=fig_size)

        try:
            yield fig, fig.subplots()

            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight", pad_inches=PAD_INCHES * style.fig_scale)

        finally:
            plt.close(fig)

    print(f"Saved: {out_path}\n")


def plot_age_scatter(
    real_ages: np.ndarray, pred_ages: np.ndarray, 
    subj_sets: list[str], subj_set_label: str, 
    N: int, slope: float, intercept: float, r: float, 
    out_path: Path, style: Style, title: str = "", 
    age_lims: tuple[int, int] = None, age_step: int = 20
):
    '''
    Plot predicted ages against the chronological ages 
    (participant sets were distinguished by marking and color)
    with an identity line and a fitted regressor.
    '''
    def _get_age_lims() -> tuple[float, float]:
        lo = min(np.nanmin(real_ages), np.nanmin(pred_ages))
        hi = max(np.nanmax(real_ages), np.nanmax(pred_ages))
        pad = (hi - lo) * .04
        return (lo - pad, hi + pad)

    age_lims = age_lims or _get_age_lims()
    note = f"N = {N}, $r$ = {r:.2f}\ny = {slope:.2f}x + {intercept:.1f}"
    note_fs = style.base_fs * .9

    with create_fig(
        style, out_path, (3.6, 3.6), (1.2, 1.2), 
        n_cols=1, n_rows=1
    ) as (fig, ax):

        ax.plot(
            age_lims, age_lims, label="identity", 
            ls="--", lw=style.base_lw, color="grey", zorder=1
        )
        for subj_set in set(subj_sets):
            mask = [ s == subj_set for s in subj_sets ]
            ax.scatter(
                real_ages[mask], pred_ages[mask], label=subj_set, 
                marker=SUBJ_SET_MARKS[subj_set][1], 
                color=SUBJ_SET_MARKS[subj_set][0], 
                alpha=.55, linewidths=0, zorder=2
            )
        ax.plot(
            age_lims, [ slope * x + intercept for x in age_lims ], 
            label=f"fit ({subj_set_label})", 
            lw=style.base_lw * 1.5, color="#333333", zorder=3
        )
        ax.annotate(
            note, xy=(1, 0), xycoords="axes fraction", 
            xytext=(note_fs * -.5, note_fs * .5), textcoords="offset points", 
            fontsize=note_fs, color="#333333", 
            bbox=dict(
                boxstyle="square,pad=.25", 
                facecolor="white", 
                edgecolor="none", 
                alpha=.75
            ), 
            ha="right", va="bottom", zorder=4
        )
        ax.legend(
            loc="upper left", 
            handlelength=1.2,  # length of the markers; default = 2, in font-size units
            borderaxespad=.5,  # the distance between the legend and the coordinate axis frame
            frameon=True, framealpha=.75, edgecolor="none"
        )
        ax.set(
            xlabel="", ylabel="", title=title, 
            xlim=age_lims, ylim=age_lims, aspect="equal"
        )
        ax.xaxis.set_major_locator(ticker.MultipleLocator(age_step))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(age_step))


def plot_coef_bars(dat: pd.Series, out_path: Path, style: Style, tip_text_gap: float = .5):
    '''
    Plot horizontal bars holding the weights of each feature in the fitted model.
    `tip_text_gap` is the space between the tip of a bar and the weight printed past it.
    '''
    def _adjust_xlim(text_axes):
        '''
        Automatically adjust the range of x-axis 
        to ensure the coef values marked above the bars are fully displayed
        and that appropriate white space is maintained around the text.
        '''
        for _ in range(2):  # ensure stability, as the initial adjustment may change the text's position
            fig.tight_layout()  # adjust the margins and spacing of subplots to avoid element overlap
            fig.draw_without_rendering()  # force a redraw of figure (with no output) to update internal state (such as text position, size, etc.)
            
            x_lo, x_hi = ax.get_xlim()
            x_offset = points_2_inches(tip_text_gap * text_fs * fig.dpi)
            coords = ax.transData.inverted().transform([(0, 0), (x_offset, 0)])  # converts pixel coordinates back into data units
            gap = np.diff(coords[:, 0]).item()  # take only the x-coordinates, calculate the difference, and convert the NumPy array to a scalar

            for text_ax in text_axes:
                bbox = text_ax.get_window_extent()  # the Axes bounding box in display space
                (text_lo, _), (text_hi, _) = ax.transData.inverted().transform(bbox)
                x_lo, x_hi = min(x_lo, text_lo - gap), max(x_hi, text_hi + gap)

            ax.set_xlim(x_lo, x_hi)

    coefs = dat.values
    y = np.arange(len(dat))
    labels = dat.index.map(model_name_2_label)
    text_fs = style.base_fs * 0.9

    with create_fig(
        style, out_path, (0.0, 0.4), (3.6, 1.3), 
        n_rows=len(dat), ytick_len=labels.str.len().max()
    ) as (fig, ax):

        ax.barh(
            y=y, width=coefs, height=0.7, 
            color=[ COEF_BAR_COLORS["+" if c >= 0 else "-"] for c in coefs ]
        )
        ax.axvline(0, color="#333333", lw=style.base_lw, zorder=3)

        text_axes = []
        for y_i, c in zip(y, coefs):
            x_pos = tip_text_gap * text_fs * (1 if c >= 0 else -1)
            text_ax = ax.annotate(
                f"{c:.2f}", xy=(c, y_i), xytext=(x_pos, 0), textcoords="offset points", 
                ha="left" if c >= 0 else "right", va="center",
                fontsize=text_fs, color="#333333"
            )
            text_axes.append(text_ax)

        ax.set_yticks(y, labels=labels)
        ax.set(xlabel="", ylabel="")
        _adjust_xlim(text_axes)


def plot_mae_bars(dat: pd.Series, dat_sd: pd.Series, out_path: Path, style: Style, tip_text_gap: float = .7):
    '''
    Plot vertical bars holding the MAE averaged across the seeds, 
    with the SD across seeds as the error bar, for each model.

    If the averaged MAE of the full-feature model is available, 
    it will be plotted as a horizontal line as a baseline.
    '''
    def _adjust_ylim(legend_ax):
        fig.tight_layout()
        fig.draw_without_rendering()

        bbox = legend_ax.get_window_extent()
        x_lo, x_hi = ax.transData.inverted().transform(bbox)[:, 0]  # in axes coordinate
        y_lo = ax.transAxes.inverted().transform(bbox)[0, 1]  # between 0 and 1

        overlapped = (x + bar_width * .5 >= x_lo) & (x - bar_width * .5 <= x_hi)
        if any(overlapped):
            dat_max = (dat + dat_sd)[overlapped].max()  # the highest point of the overlapped bars, in data unit
            new_y_hi = dat_max / (y_lo - 0.03)          # make it coincide with the bottom of the legend (with a small gap)
            ax.set_ylim(top=max(ax.get_ylim()[1], new_y_hi))

    mae = dat.values
    mae_sd = dat_sd.values
    x = np.arange(len(dat))
    labels = dat.index.map(model_name_2_label)
    modes = dat.index.str.split("_").str[0]
    baseline = dat.get("full", None)
    text_fs = style.base_fs * 0.9
    bar_width = 0.7
    
    with create_fig(
        style, out_path, (0.6, 0.0), (1.5, 4), 
        n_cols=len(dat), xtick_len=labels.str.len().max(), 
    ) as (fig, ax):

        ax.bar(
            x, mae, yerr=mae_sd, width=bar_width, 
            color=[ MODEL_MODES.get(m, ("", "grey"))[1] for m in modes ], 
            error_kw=dict(
                elinewidth=style.base_lw * 1.5, ecolor="#333333"
            )
        )
        for x_i, v in zip(x, mae):
            y_pos = tip_text_gap * text_fs * -1
            text_ax = ax.annotate(
                f"{v:.2f}", xy=(x_i, v), xytext=(0, y_pos), textcoords="offset points", 
                rotation=90, va="top", ha="center",
                fontsize=text_fs, color="white"
            )

        handles = []
        for mode, (label, color) in MODEL_MODES.items():
            if mode in modes:
                handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))
        
        if baseline is not None:
            baseline_ax = ax.axhline(
                baseline, label=f"{MODEL_MODES['full'][0]} = {baseline:.2f}", 
                color=MODEL_MODES["full"][1], lw=style.base_lw, ls="--", zorder=3
            )
            handles.append(baseline_ax)

        legend_ax = ax.legend(
            handles=handles, title="Feature set", frameon=False, loc="upper left"
        )
        ax.set_xticks(x, labels=labels, rotation=90, ha="center", va="top")
        ax.set(xlabel="", ylabel="")
        _adjust_ylim(legend_ax)


def plot_corr_boxes(
    df: pd.DataFrame, 
    model_order: list[str], 
    model_stars: pd.Series, 
    fdr_alpha: float, 
    out_path: Path, 
    style: Style
):
    '''
    Plot boxplots holding the distribution of correlations coefficients
    between a model's PAD(ac) values and individual scores, one per model, in `model_order`.

    Points that survive FDR correction are outlined.
    Models whose correlations are jointly displaced from zero (FDR-corrected) 
    are marked with stars (based on `model_stars`).
    '''
    modes = df["Model"].str.split("_").str[0]
    mode_order = [ m for m in MODEL_MODES.keys() if (modes == m).any() ]
    mode_palette = { m: MODEL_MODES[m][1] for m in mode_order }
    df["Mode"] = modes

    domain_ordar = sorted(df["Domain"].unique())
    domain_palette = dict(zip(domain_ordar, sns.color_palette(n_colors=len(domain_ordar))))

    model_labels = [ model_name_2_label(m) for m in model_order ]
    sig_edge_width = 0.8 * style.fig_scale
    r_max = df["r"].max()

    with create_fig(
        style, out_path, (0.5, 0.0), (2.5, 5), 
        n_cols=len(model_order), 
        xtick_len=max(map(len, model_labels))
    ) as (fig, ax):

        sns.boxplot(
            data=df, x="Model", y="r", order=model_order, ax=ax, 
            hue="Mode", hue_order=mode_order, palette=mode_palette, 
            fill=False, dodge=False, width=0.5, fliersize=0, legend=False
        )
        for is_sig, sub_df in df.groupby("q_sig", observed=True):
            sns.stripplot(
                data=sub_df, x="Model", y="r", order=model_order, ax=ax, 
                hue="Domain", hue_order=domain_ordar, palette=domain_palette, 
                edgecolor="black", linewidth=sig_edge_width if is_sig else 0, 
                jitter=.2, dodge=False, alpha=.9 if is_sig else .5, legend=False
            )
        ax.axhline(
            0, color="grey", lw=style.base_lw, ls="--", zorder=0
        )

        fig.tight_layout()
        fig.draw_without_rendering()
        y_lo, y_hi = ax.get_ylim()
        y_pos = r_max + ((y_hi - y_lo) * .03)  # with a small gap

        for x_i, model in enumerate(model_order):
            star = model_stars[model]
            ax.text(
                x_i, y_pos, star, ha="center", va="bottom", 
                fontsize=style.base_fs if star != NS_LABEL else style.base_fs * .8, 
                fontweight="bold" if star != NS_LABEL else "normal",
                color="black" if star != NS_LABEL else "grey"
            )
        ax.set_ylim(
            y_lo - ((y_hi - y_lo) * .25),  # leave space for legend rows
            max(y_hi, y_pos + ((y_hi - y_lo) * .09))
        )
        # y_lo, y_hi = ax.get_ylim()
        # y_step = round((y_hi - y_lo) / 5, 2)
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))

        box_handles = [
            plt.Line2D(
                [], [], label=MODEL_MODES[m][0], 
                lw=style.base_lw * 2, color=mode_palette[m]
            ) for m in mode_order
        ]
        dot_handles = [
            plt.Line2D(
                [], [], label=capitalize_upper_label(d), 
                ls="", marker="o", color=domain_palette[d]
            ) for d in domain_ordar
        ]
        dot_handles.append(
            plt.Line2D(
                [], [], label=f"$q$ < {fprint_prob(fdr_alpha)}", 
                ls="", marker="o", color="lightgrey", 
                markeredgecolor="black", markeredgewidth=sig_edge_width
            )
        )
        n_cols = max(len(box_handles), len(dot_handles))
        handles = [
            group[i] if i < len(group) else plt.Line2D([], [], ls="", label="")
            for i in range(n_cols) for group in (box_handles, dot_handles)
        ]
        ax.legend(
            handles=handles, title="", frameon=False,
            ncol=n_cols, loc="lower center"
        )

        ax.set_xticks(
            range(len(model_order)), model_labels, 
            rotation=90, ha="center", va="top"
        )
        ax.set(xlabel="", ylabel="")


def plot_corr_heat(
    df: pd.DataFrame,
    model_order: list[str], 
    score_key: str, 
    fdr_alpha: float, 
    out_path: Path,
    style: Style, 
    annot_mode: str = "star", 
    xticks_rotate: int = 45
):
    '''
    Plot the same correlations `plot_corr_boxes` collapses into boxes 
    as a model x score (grouped by domain) matrix, 
    with color holding r and cell labels carry q-values or stars.
    '''
    def _matrix(value_col: str) -> pd.DataFrame:
        mat = df.pivot(index="Model", columns="Score", values=value_col)
        return mat.reindex(index=model_order, columns=score_order)

    score_df = df.loc[:, ["Domain", "Score"]].sort_values("Score").drop_duplicates()
    score_order = score_df["Score"].to_list()
    score_labels = [ capitalize_upper_label(s.split(score_key)[-1] if score_key else s) for s in score_order ]
    model_labels = [ model_name_2_label(m) for m in model_order ]

    domains = score_df["Domain"].to_list()
    domain_boundaries = [0] + [ i for i in range(1, len(domains)) if domains[i] != domains[i - 1] ] + [len(domains)]
    domain_fs = style.base_fs * 1.1

    r_mat, q_mat = _matrix("r"), _matrix("q")
    star_mat = q_mat.map(lambda q: sig_stars(q, fdr_alpha)).replace(NS_LABEL, "")
    annot_mat = {
        "r": [
            [ "" if not np.isfinite(r) else f"{r:.2f}\n{s}".strip() 
              for r, s in zip(r_row, star_row) ]
            for r_row, star_row in zip(r_mat.to_numpy(), star_mat.to_numpy()) 
        ], 
        "star": star_mat
    }[annot_mode]
    r_lim = float(np.nanmax(np.abs(r_mat.to_numpy()))) if r_mat.notna().any().any() else 1

    with create_fig(
        style, out_path, (0.4, 0.4), (1, 2), 
        n_cols=len(score_order), n_rows=len(model_order), 
        ytick_len=max(map(len, model_labels)), 
        xtick_len=max(map(len, score_labels)) * sin(radians(xticks_rotate))
    ) as (fig, ax):

        sns.heatmap(
            r_mat, ax=ax, fmt="", annot=annot_mat, # annot_kws={"fontsize": }, 
            cmap="vlag", center=0, vmin=-r_lim, vmax=r_lim, 
            cbar_kws=dict(
                fraction=.035, pad=.015, shrink=.45, aspect=22
            ), 
            linewidths=style.base_lw * 0.5, linecolor="white", 
            square=True if annot_mode == "star" else False
        )
        for x_i in domain_boundaries[1:-1]:
            ax.axvline(
                x_i, color="#333333", lw=style.base_lw * 1.2
            )
        for lo, hi in zip(domain_boundaries[:-1], domain_boundaries[1:]):
            ax.annotate(
                capitalize_upper_label(domains[lo]), 
                xy=((lo + hi) / 2, 1), xycoords=("data", "axes fraction"), 
                xytext=(0, domain_fs * 0.4), textcoords="offset points", 
                ha="center", va="bottom", fontsize=domain_fs, 
                fontstyle="italic", # fontweight="bold"
            )

        ax.set_xticks(
            np.arange(len(score_order)) + .5, score_labels, 
            ha="right", va="top", rotation=xticks_rotate, rotation_mode="anchor"
        )
        ax.set_yticks(
            np.arange(len(model_order)) + .5, model_labels, 
            ha="right", va="center", rotation=0
        )
        ax.set(xlabel="", ylabel="")
    

