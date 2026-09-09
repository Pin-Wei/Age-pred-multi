#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from helpers import print_missing


class Config:
    def __init__(self):
        proj_root = Path(__file__).resolve().parents[1]
        self.eeg_data_dir    = proj_root / "data" / "rs-eeg"
        self.eeg_data_paths  = self.eeg_data_dir.glob("sub-*_task-rest.set")
        self.df_preproc_path = proj_root / "data" / "tabular" / "df_preproc.csv"
        self.tbl_out_path    = proj_root / "data" / "tabular" / "df_rs-eeg_psd.csv"
        self.npz_out_path    = proj_root / "data" / "tabular" / "df_rs-eeg_psd.npz"


def compute_psd(data_path: Path) -> tuple[np.ndarray, list[str]]:
    '''
    For each .set file:
    - Load with mne.io.read_raw_eeglab
    - Compute power spectral density with Welch's method (4s window, 50% overlap, hann)
    - Average log-power per canonical band: 
        delta (1-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-45) Hz
    - Output: (n_channels=30) × (n_bands=5) = 150 features per subject
    '''
    raw = mne.io.read_raw_eeglab(str(data_path), preload=True, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    spec = raw.compute_psd(
        method="welch", fmin=1.0, fmax=45.0, 
        n_fft=int(4 * sfreq), n_overlap=int(2 * sfreq), 
        verbose="ERROR"
    )
    psd, freqs = spec.get_data(return_freqs=True)  # (n_ch, n_freqs)

    feats = []
    feat_names = []  # (n_ch * n_bands,)

    for band, (lo, hi) in {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_pow = psd[:, mask].mean(axis=1)  # (n_ch,)
        band_pow = np.log10(band_pow + 1e-30)
        feats.append(band_pow)
        feat_names.extend([f"EEG_RESTING_{band}_{ch}" for ch in ch_names])

    return pd.Series(np.concatenate(feats), index=feat_names)


def main():
    config = Config()

    subj_df = pd.read_csv(config.df_preproc_path, usecols=["BASIC_INFO_ID"])
    subj_list = subj_df["BASIC_INFO_ID"].tolist()
    n_subjs = len(subj_list)
    print(f"\nNumber of participants: {n_subjs}")
    
    subj_files = { p.name.split("_")[0]: p for p in config.eeg_data_paths }
    overlap = [ s for s in subj_list if s in subj_files.keys() ]
    print(f"EEG .set files available: {len(subj_files)}\n")

    feat_names = None
    X_dict = {}

    for n, subj in enumerate(overlap, 1):
        try:
            values = compute_psd(subj_files[subj])
            X_dict[subj] = values
            print(f"[{n:04d} / {len(overlap)}] Successfully processed data for {subj}")

            if feat_names is None:
                feat_names = values.index.tolist()

            elif set(values.index) != set(feat_names):
                extra = sorted(set(values.index) - set(feat_names))
                absent = sorted(set(feat_names) - set(values.index))
                print(f"\n[Warning] {subj} has a different channel set than the reference:")
                print(f"\tunexpected: {extra}")
                print(f"\tmissing: {absent}")

        except Exception as e:
            print(f"\nError occurred when processing data for {subj}:\n{e}\n")

    print_missing(subj_list, X_dict.keys())

    if not X_dict:
        raise RuntimeError("No EEG file could be processed; refusing to write an empty table.")

    X_df = pd.DataFrame(X_dict).T  # build DataFrame from Series to ensure feature names are aligned
    X_df = X_df.reindex(columns=feat_names)  # canonical order; absent -> NaN
    
    X = X_df.to_numpy()
    np.savez(config.npz_out_path, X=X, SID=X_df.index.to_numpy(), feats=np.array(feat_names))
    print(f"\nWrote {X.shape[0]} rows × {X.shape[1]} features into:")
    print(f"- NPZ file: {config.tbl_out_path}")

    out_df = pd.merge(
        subj_df.rename(columns={"BASIC_INFO_ID": "SID"}), 
        X_df.reset_index(names="SID"), 
        on="SID", how="left"
    )
    out_df.to_csv(config.tbl_out_path, index=False)
    print(f"- CSV file: {config.tbl_out_path}\n")


if __name__ == "__main__":
    mne.utils.set_config('MNE_USE_CUDA', 'true')
    main()
    