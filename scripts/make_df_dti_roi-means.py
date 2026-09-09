#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.datasets import fetch_atlas_juelich
from nilearn.image import resample_to_img

from helpers import print_missing, load_img_data


class Config:
    def __init__(self):
        self.atlas_name = "maxprob-thr25-1mm"

        proj_root = Path(__file__).resolve().parents[1]
        tbss_dir             = proj_root / "data" / "dti" / "tbss"
        self.fa_3d_dir       = tbss_dir / "origdata"
        self.fa_3d_paths     = sorted(self.fa_3d_dir.glob("*.nii.gz"))
        self.fa_4d_path      = tbss_dir / "stats" / "all_FA.nii.gz"
        self.md_4d_path      = tbss_dir / "stats" / "all_MD.nii.gz"
        self.df_preproc_path = proj_root / "data" / "tabular" / "df_preproc.csv"
        self.tbl_out_path    = proj_root / "data" / "tabular" / "df_dti_roi-means.csv"
        self.npz_out_path    = proj_root / "data" / "tabular" / "df_dti_roi-means.npz"


def get_roi_means(img_path: str, N: int, atlas_name: str = None, atlas_infos: tuple[np.asarray, list[int], list[str]] = None):
    def _get_atlas_infos(atlas_name, img_grid, img_affine):
        '''
        - Fetch the Juelich max-probability atlas
        - Resample it (nearest-neighbour) onto the reference grid
        - Returns the integer label volume, the non-background label ids, and their names
        '''
        print(f"Loading infos of atlas '{atlas_name}' ...")

        atlas = fetch_atlas_juelich(atlas_name=atlas_name)
        ref = nib.Nifti1Image(np.zeros(img_grid, dtype=np.int8), img_affine)
        atlas_aligned = resample_to_img(
            atlas["maps"], ref, interpolation="nearest", 
            force_resample=True, copy_header=True,
        )
        label_vol = np.asarray(atlas_aligned.dataobj, dtype=np.int16)

        labels = list(atlas["labels"])  # index 0 == "Background"
        label_ids = [ i for i in np.unique(label_vol) if i > 0 ]
        label_names = [ labels[i].strip().replace(" ", "") for i in label_ids ]

        return (label_vol, label_ids, label_names)

    img_dat, img_affine = load_img_data(img_path, get_affine=True)
    N_v = img_dat.shape[-1]
    assert N_v == N, f"Mismatch between volume ({N_v}) and globbed ({N}) subject count."

    img_flat = img_dat.reshape(-1, N)  # (n_vox, N)

    if atlas_infos is None:
        assert atlas_name is not None, "If 'atlas_infos' is not given, 'atlas_name' should be provided."
        img_grid = img_dat.shape[:3]
        atlas_infos = _get_atlas_infos(atlas_name, img_grid, img_affine)

    label_vol, label_ids, label_names = atlas_infos
    label_vol_flat = label_vol.reshape(-1)  # (n_vox,)

    n_rois = len(label_ids)
    roi_means = np.full((N, n_rois), np.nan, dtype=np.float32)
    is_empty = []

    for i, idx in enumerate(label_ids):
        roi = img_flat[label_vol_flat == idx]  # (n_vox_roi, N)
        mask = roi > 0
        count = mask.sum(axis=0)
        roi_mean = (roi * mask).sum(axis=0) / np.maximum(count, 1)
        roi_means[:, i] = np.where(count > 0, roi_mean, np.nan)

        if not count.all():
            n_empty = int((count == 0).sum())
            is_empty.append((label_names[i], n_empty))

    if is_empty:
        print(f"\n[Warning] {len(is_empty)} ROI(s) had no in-brain voxel for some participant(s):")
        for roi_name, n_bad in empty:
            print(f"\t- {roi_name}: {n_bad} participant(s) -> NaN")

    return roi_means, atlas_infos


def main():
    config = Config()

    subj_df = pd.read_csv(config.df_preproc_path, usecols=["BASIC_INFO_ID"])
    subj_list = subj_df["BASIC_INFO_ID"].tolist()
    n_subjs = len(subj_list)
    print(f"\nNumber of participants: {n_subjs}")

    dti_subjs = [ fp.name.split(".")[0] for fp in config.fa_3d_paths ]  # the order TBSS merge per-subject FA volumes
    print_missing(subj_list, dti_subjs)

    print("\nCalculating ROI means for FA data ...")
    X_fa, atlas_infos = get_roi_means(
        img_path=config.fa_4d_path, 
        N=len(dti_subjs), 
        atlas_name=config.atlas_name
    )

    print("\nCalculating ROI means for MD data ...")
    X_md, _ = get_roi_means(
        img_path=config.md_4d_path, 
        N=len(dti_subjs), 
        atlas_infos=atlas_infos
    )

    label_names = atlas_infos[-1]
    feat_names = (
        [ f"DTI_FA_{n}" for n in label_names ] + 
        [ f"DTI_MD_{n}" for n in label_names ]
    )
    n_feat = len(feat_names)

    X = np.concatenate([X_fa, X_md], axis=1)
    np.savez(config.npz_out_path, X=X, SID=np.array(dti_subjs), feats=np.array(feat_names))
    print(f"\nWrote {X.shape[0]} rows × {X.shape[1]} features into:")
    print(f"- NPZ file: {config.tbl_out_path}")
    
    X_df = pd.DataFrame(X, columns=feat_names)
    X_df.insert(0, "SID", dti_subjs)
    
    out_df = pd.merge(
        subj_df.rename(columns={"BASIC_INFO_ID": "SID"}), X_df, 
        on="SID", how="left"
    )
    out_df.to_csv(config.tbl_out_path, index=False)
    print(f"- CSV file: {config.tbl_out_path}\n")


if __name__ == "__main__":
    main()
