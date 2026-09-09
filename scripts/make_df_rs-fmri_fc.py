#!/usr/bin/env python3

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018

from helpers import print_missing


class Config:
    def __init__(self):
        self.TR = 2.4
        self.n_rois = 400
        self.conf_cols = (
            [ f"trans_{p}" for p in ["x", "y", "z"] ] + 
            [ f"rot_{p}" for p in ["x", "y", "z"] ]
        )
        deri_cols = [ f"{c}_derivative1" for c in self.conf_cols ]
        self.conf_cols += deri_cols
        self.conf_cols += [ f"a_comp_cor_{i:02d}" for i in range(5) ]
        self.conf_cols += [ f"cosine{i:02d}" for i in range(6) ] 

        proj_root = Path(__file__).resolve().parents[1]
        self.mri_data_dir    = proj_root / "data" / "rs-mri"
        self.df_preproc_path = proj_root / "data" / "tabular" / "df_preproc.csv"
        self.tbl_out_path    = proj_root / "data" / "tabular" / "df_rs-fmri_fc.csv"
        self.npz_out_path    = proj_root / "data" / "tabular" / "df_rs-fmri_fc.npz"


def compute_fc(subj: str, bold_path: str, mask_path: str, conf_path: str, atlas_path: str, config: Config):
    '''
    Per subject:
    - NiftiLabelsMasker with given atlas (Schaefer 2018, 400 ROIs, 7 networks)
    - Confound denoising
    - Standardization + detrend + 0.008-0.08 Hz bandpass
    - Extract ROI timeseries → Fisher-z Pearson FC matrix
    - Vectorize upper triangle → 79,800 edges per subject 
    '''
    try:
        conf = pd.read_csv(conf_path, sep="\t")
        use_cols = [ c for c in config.conf_cols if c in conf.columns ]
        C = conf[use_cols].fillna(0.0).to_numpy(dtype=np.float32)

        masker = NiftiLabelsMasker(
            labels_img=atlas_path, 
            mask_img=mask_path, 
            standardize="zscore_sample", 
            detrend=True, 
            low_pass=0.08, 
            high_pass=0.008, 
            t_r=config.TR, 
            memory=None, 
            verbose=0
        )
        ts = masker.fit_transform(str(bold_path), confounds=C)  # (n_TRs, n_labels_found)
        
        region_map = masker.region_ids_  # {column_index_in_ts: schaefer_label_value_as_float}
        cids = np.fromiter(( k for k in region_map.keys() if k != "background" ), dtype=int)
        lids = np.fromiter(( int(region_map[k]) for k in cids ), dtype=int)
        valid_lid_idx = (lids >= 1) & (lids <= config.n_rois) 
        source_cols = cids[valid_lid_idx]
        target_cols = lids[valid_lid_idx] - 1  # convert to zero-based index
        ts_full = np.full((ts.shape[0], config.n_rois), np.nan, dtype=np.float32)  # (n_TRs, n_rois)
        ts_full[:, target_cols] = ts[:, source_cols]

        is_all_nan = np.isnan(ts_full).all(axis=0)  # identify which ROIs have all TRs that are NaN
        valid_roi_idx = np.where(~is_all_nan)[0]
        ts_valid = ts_full[:, valid_roi_idx]

        fc_valid = np.corrcoef(ts_valid, rowvar=False)  # each column represents a variable; (n_rois, n_rois)
        
        fc_valid = np.clip(fc_valid, -1 + 1e-7, 1 - 1e-7)
        fcz_valid = np.arctanh(fc_valid)  # convert to Fisher Z-scores

        fcz_full = np.full((config.n_rois, config.n_rois), np.nan)
        idx_mesh = np.ix_(valid_roi_idx, valid_roi_idx)
        fcz_full[idx_mesh] = fcz_valid

        iu = np.triu_indices(config.n_rois, k=1)  # upper triangle indices, excluding diagonal
        edges = fcz_full[iu].astype(np.float32)  # flatten

        return (subj, edges, "ok")

    except Exception as e:
        return (subj, None, f"{type(e).__name__}: {str(e)[:120]}")


def main():
    config = Config()

    atlas = fetch_atlas_schaefer_2018(n_rois=config.n_rois, yeo_networks=7, resolution_mm=2)
    atlas_path = atlas["maps"]
    print(f"Atlas: {atlas_path}")

    subj_df = pd.read_csv(config.df_preproc_path, usecols=["BASIC_INFO_ID"])
    subj_list = subj_df["BASIC_INFO_ID"].tolist()
    n_subjs = len(subj_list)
    print(f"\nNumber of participants: {n_subjs}")

    subj_files = {}

    for subj in subj_list:
        bold = config.mri_data_dir / f"{subj}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        mask = config.mri_data_dir / f"{subj}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        conf = config.mri_data_dir / f"{subj}_ses-01_task-rest_desc-confounds_timeseries.tsv"
        
        if bold.exists() and mask.exists() and conf.exists():
            subj_files[subj] = {
                "bold_path": str(bold), 
                "mask_path": str(mask), 
                "conf_path": str(conf), 
                "atlas_path": str(atlas_path)
            }

    print_missing(subj_list, subj_files.keys())

    iu = np.triu_indices(config.n_rois, k=1) 
    edge_cols = [f"fc_{i + 1:03d}_{j + 1:03d}" for i, j in zip(iu[0], iu[1])]
    n_edges = len(edge_cols)

    X = np.full((n_subjs, n_edges), np.nan, dtype=np.float32)
    n_done = 0

    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(compute_fc, subj=k, **v, config=config): k 
            for k, v in subj_files.items()
        }
        
        for future in as_completed(futures):
            subj, edges, msg = future.result()
            n_done += 1
            prog = f"[{n_done:04d} / {len(subj_files):04d}]"

            if edges is not None:
                idx = subj_list.index(subj)
                X[idx] = edges
                print(f"{prog} Successfully processed data for {subj}")
            else:
                print(f"\n{prog} Failed to process data for {subj}:\n{msg}\n", flush=True)

    np.savez(config.npz_out_path, X=X, SID=np.array(subj_list), feats=np.array(edge_cols))
    print(f"\nWrote {X.shape[0]} rows × {X.shape[1]} features into:")
    print(f"- NPZ file: {config.tbl_out_path}")

    out_df = pd.DataFrame(X, columns=edge_cols)
    out_df.insert(0, "SID", subj_list)
    out_df.to_csv(config.tbl_out_path, index=False)
    print(f"- CSV file: {config.tbl_out_path}\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()