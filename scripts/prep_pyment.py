#!/usr/bin/env python3

from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import nibabel as nib


class Config:
    def __init__(self):
        root_dir = Path(__file__).resolve().parents[1]
        self.subj_df_path = root_dir / "data" / "tabular" / "df_preproc.csv"
        self.img_src_root = root_dir / ".." / "Age_pred_DL" / "data" / "freesurfer"
        self.img_dst_dir = root_dir / "data" / "pyment"
        self.out_csv_path = root_dir / "data" / "meta" / "pyment_labels.csv"


def binarize_mask(mask_path: Path, output_path: Path):
    '''
    Binarize a mask image and save it to the output path.
    '''
    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata()
    binarized_data = (mask_data > 0).astype(np.uint8)
    binarized_img = nib.Nifti1Image(binarized_data, affine=mask_img.affine, header=mask_img.header)
    nib.save(binarized_img, str(output_path))


def main():
    config = Config()

    subj_df = pd.read_csv(
        config.subj_df_path, 
        index_col="BASIC_INFO_ID", 
        usecols=["BASIC_INFO_ID", "BASIC_INFO_AGE", "BASIC_INFO_SEX"]
    )

    ## Copy certain FreeSurfer outputs to the destination directories
    for sid in subj_df.index:
        print(f"Copying images for subject {sid} ...")
        src_dir = config.img_src_root / sid / "mri"

        ## T1 image
        dst = config.img_dst_dir / "raw" / f"{sid}.nii.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_dir / "T1.mgz", dst)

        ## Binarized brain mask & other fastsurfer outputs (see: https://github.com/Deep-MI/FastSurfer/blob/dev/doc/overview/OUTPUT_FILES.md#segmentation-module)
        dst_dir = config.img_dst_dir / "fastsurfer" / sid / "mri"
        dst_dir.mkdir(parents=True, exist_ok=True)
        binarize_mask(src_dir / "brainmask.mgz", dst_dir / "mask.mgz")
        for fn in [
            "aparc.DKTatlas+aseg.mgz", 
            "aseg.auto_noCCseg.mgz", 
            "orig.mgz", 
            "orig_nu.mgz"
        ]:
            if fn == "aparc.DKTatlas+aseg.mgz":
                shutil.copyfile(src_dir / fn, dst_dir / "aparc.DKTatlas+aseg.deep.mgz")
            else:
                shutil.copyfile(src_dir / fn, dst_dir / fn)

    ## Prepare the CSV file with the required columns for pyment (i.e., image_id, age, sex, diagnosis, has_diagnosis; see: https://github.com/estenhl/pyment-public/blob/main/scripts/download_ds000030.py)
    subj_df.insert(0, "image_id", subj_df.index)
    subj_df.rename(columns={
        "BASIC_INFO_AGE": "age", 
        "BASIC_INFO_SEX": "sex"
    }, inplace=True)
    subj_df["diagnosis"] = "CONTROL"
    subj_df["has_diagnosis"] = subj_df["diagnosis"] != "CONTROL"
    subj_df.to_csv(config.out_csv_path, index=False)


if __name__ == "__main__":
    main()
    print("\nFinished preparing files required for running Pyment.\n")