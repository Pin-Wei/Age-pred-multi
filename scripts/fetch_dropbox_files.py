#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

from dotenv import load_dotenv
import dropbox


class Config:
    def __init__(self):
        self.dropbox_refresh_token = os.getenv("DROPBOX_REFRESH_TOKEN")
        self.dropbox_app_key = os.getenv("DROPBOX_APP_KEY")
        self.dropbox_app_secret = os.getenv("DROPBOX_APP_SECRET")
        self.dropbox_eeg_path_template = "/「tcnl-quanta 」團隊資料夾/analysis/EEG_rest/Resting_EEG/microstateLAB_Data/{}_ses-01_task-rest.set" 
        self.dropbox_fmri_dir_template = "/「tcnl-quanta 」團隊資料夾/fmriprep_out/fmriprep/{}/ses-01/func"
        self.dropbox_fmri_file_templates = [
            "{}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
            "{}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
            "{}_ses-01_task-rest_desc-confounds_timeseries.tsv"
        ]
        local_data_root = Path(__file__).resolve().parents[1] / "data" 
        self.local_eeg_dir   = local_data_root / "rs-eeg"
        self.local_mri_dir   = local_data_root / "rs-mri"
        self.df_preproc_path = local_data_root / "tabular" / "df_preproc.csv"


def fetch_file(local_path: Path, dropbox_path: str, dbx: dropbox.Dropbox, msg: str):
    if local_path.is_file():
        if local_path.stat().st_size > 0:
            print(f"[{msg}] File '{local_path.name}' exists. Skipping ...")
            return True

        print(f"[{msg}] File '{local_path.name}' is 0 KB. Deleting and re-downloading ...")
        local_path.unlink()

    try:
        with open(local_path, "wb") as f:
            metadata, resp = dbx.files_download(path=dropbox_path)
            f.write(resp.content)

        size_mb = local_path.stat().st_size / 1e6
        print(f"[{msg}] Downloaded file: {local_path.name} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n[Error] Failed to fetch file: {os.path.basename(dropbox_path)}")
        print(f"{e}\n")
        local_path.unlink()
        return False


def main():
    config = Config()

    try:
        dbx = dropbox.Dropbox(
            oauth2_refresh_token=config.dropbox_refresh_token,
            app_key=config.dropbox_app_key,
            app_secret=config.dropbox_app_secret
        )
        print(f"\nConnected to Dropbox account: {dbx.users_get_current_account().email}")
        
        ## Switch from home namespace to root(team) namespace 
        root_ns = dbx.users_get_current_account().root_info.root_namespace_id
        dbx = dbx.with_path_root(dropbox.common.PathRoot.root(root_ns))

        subj_df = pd.read_csv(config.df_preproc_path, usecols=["BASIC_INFO_ID"])
        subj_list = subj_df["BASIC_INFO_ID"].tolist()
        missing = []

        print(f"\n=== Downloading EEG files ===")
        config.local_eeg_dir.mkdir(parents=True, exist_ok=True)
        c = 1
        tot = len(subj_list)
        for subj in subj_list:
            dropbox_path = config.dropbox_eeg_path_template.format(subj)
            file = os.path.basename(dropbox_path)
            local_path = config.local_eeg_dir / file
            if not fetch_file(local_path, dropbox_path, dbx, msg=f"{c:04d} / {tot:04d}"):
                missing.append(dropbox_path)
            c += 1

        print(f"\n=== Downloading rsfMRI files ===")
        config.local_mri_dir.mkdir(parents=True, exist_ok=True)
        c = 1
        tot = len(subj_list) * len(config.dropbox_fmri_file_templates)
        for subj in subj_list:
            for file in config.dropbox_fmri_file_templates:
                dropbox_path = config.dropbox_fmri_dir_template.format(subj) + "/" + file.format(subj)
                local_path = config.local_mri_dir / file.format(subj)
                if not fetch_file(local_path, dropbox_path, dbx, msg=f"{c:04d} / {tot:04d}"):
                    missing.append(dropbox_path)
                c += 1

        if missing:
            print(f"\n[Warning] {len(missing)} file(s) not found on Dropbox:")
            for m in missing:
                print(f"\t- {m}")

        print("\nDone.\n")

    except Exception as e:
        print(f"\nError occurred:\n{e}\n")


if __name__ == "__main__":  
    load_dotenv()
    main()