#!/usr/bin/env python3


import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import custom_print


def print_missing(all_subjs: list[str], data_subjs: list[str]) -> list[str]:
    '''
    Print and return the sorted list of participants 
    that present in `all_subjs` (the reference roster)
    but absent from `data_subjs` (the participants a given modality actually has data for).
    '''
    missing = sorted(set(all_subjs).difference(set(data_subjs)))

    if missing:
        custom_print(
            f"\ndata of {len(missing)} participant(s) are missing:", 
            level="WARNING"
        )
        for m in missing:
            custom_print(f"\t- {m}")

    return missing


def load_img_data(
    img_path: str | Path,
    print_infos: bool = False,
    get_affine: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    '''
    Load a NIfTI image from `img_path`
    and return its voxel data as a float32 array (with the image's own shape).

    If `print_infos` is set to True, 
    also print the image shape, axis orientation codes, etc.

    If `get_affine` is set to True, 
    also return the 4x4 voxel-to-world affine.
    '''
    
    import nibabel as nib

    def _print_infos(img: nib.nifti1.Nifti1Image):
        hdr = img.header
        custom_print(f"Image shape: {img.shape}")
        custom_print(f"Image orientations: {nib.aff2axcodes(img.affine)}")
        custom_print({
            0: "sform not defined", 
            1: "RAS+ in scanner coordinates", 
            2: "RAS+ aligned to some other scan", 
            3: "RAS+ in Talairach atlas space", 
            4: "RAS+ in MNI atlas space"
        }[int(hdr["sform_code"])])
        custom_print(f'Description: {hdr["descrip"]}')
        custom_print()

    img_path = str(img_path)
    assert os.path.isfile(img_path), f"Image file not exists: {img_path}"

    custom_print(f"Loading image: {img_path}")
    img = nib.load(img_path)

    if print_infos:
        _print_infos(img)

    custom_print("Loading the array data ...")
    img_dat = np.asarray(img.dataobj, dtype=np.float32)  # avoid caching
        # see: https://nipy.org/nibabel/images_and_memory.html#use-the-array-proxy-instead-of-get-fdata

    if get_affine:
        return img_dat, img.affine
    else:
        return img_dat


def get_tbss_processed(
    img_path: str | Path, 
    subj_dir: str | Path,
    stride: int, 
    mask_path: str | Path,
    cache: str | Path
) -> tuple[list[str], np.ndarray]:
    '''
    List the .nii.gz files under `subj_dir` and sort their names (participant IDs), 
    which should be the order TBSS merge per-participant FA volumes.

    Load the 4-D TBSS stack from `img_path`, 
    downsample all three spatial axes by `stride`,
    keep the voxels whose value > 0 in the 3-D mask `mask_path`
    and flatten them per participant.

    Return the participant IDs and the flattened (N, n_vox) array.

    If a processed NPY file (`cache`) exists
    and its sibling JSON file's key matches, reuse it;
    otherwise, rebuilt and save them (cache and key).

    Note: No scaling is applied here; 
    `train_eval_model` fits a `StandardScaler` per fold on its own training subset.
    '''
    def _cache_key():
        '''
        The identity of the result, which includes:
        - The file name and byte size of the source and mask images
        - `stride`
        - The participant IDs, in the order of the volumes
        '''
        def _stat(p):
            p = str(p)
            return {"name": os.path.basename(p), "bytes": os.path.getsize(p)}

        return {
            "img"   : _stat(img_path),
            "mask"  : _stat(mask_path),
            "stride": int(stride),
            "SID"   : subj_list
        }

    cache = Path(cache)
    if cache.suffix != ".npy": 
        cache = cache.with_name(cache.name + ".npy")
        
    key_path = cache.with_suffix(".json")
    subj_list = [ fp.name.split(".")[0] for fp in sorted(Path(subj_dir).glob("*.nii.gz")) ]
    has_sources = bool(subj_list) and os.path.isfile(img_path) and os.path.isfile(mask_path)

    if cache.exists():
        try:
            cached_key = json.loads(key_path.read_text())
        except (OSError, ValueError):
            cached_key = None

        if not has_sources:
            msg1 = "\nThe source images are not found, "
            msg2 = f"\nRebuild {cache.name} where the source images are.\n"
            if not cached_key:
                raise FileNotFoundError(msg1 + "nor a readable cache key exists." + msg2)
            if "SID" not in cached_key:
                raise ValueError(msg1 + f"and {key_path.name} holds no participant IDs." + msg2)
            if cached_key.get("stride") != int(stride):
                raise ValueError(msg1 + f"yet the cache is built with a stride size of {cached_key.get('stride')}." + msg2)
            custom_print(
                f"Loaded from cache, without the source images to check it against: {cache}", 
                level="WARNING"
            )
            return cached_key["SID"], np.load(cache, mmap_mode="r")

        key = _cache_key()
        if cached_key == key:
            custom_print(f"Loaded from cache: {cache}")
            return subj_list, np.load(cache, mmap_mode="r")

        stale = (
            "no readable cache key" if cached_key is None 
            else ", ".join(
                ("participant IDs" if k == "SID" else f"{k}: {cached_key.get(k)!r} -> {v!r}") 
                for k, v in key.items() 
                if cached_key.get(k) != v
            )
        )
        custom_print(
            f"Stale cache, recomputing ({stale}): {cache}", 
            level="WARNING"
        )

    else:
        if not has_sources:
            raise FileNotFoundError(
                f"\nCannot build {cache.name}; the source images are not found "
                f"(stack: {img_path}, mask: {mask_path}, per-participant images: {subj_dir})\n"
            )
        key = _cache_key()

    img_dat = load_img_data(img_path)
    N = img_dat.shape[-1]
    if N != len(subj_list):
        raise ValueError(f"Mismatch between volume ({N}) and globbed ({len(subj_list)}) subject count.")
    
    img_dat = img_dat[::stride, ::stride, ::stride, :]
    img_dat = np.moveaxis(img_dat, -1, 0)  # (N, X, Y, Z)

    mask_dat = load_img_data(mask_path)
    mask_dat = mask_dat[::stride, ::stride, ::stride]
    bin_mask = mask_dat > 0

    img_flat = img_dat[:, bin_mask].copy()  # (N, n_vox)

    if cache.exists():  # keep the old cache and its key
        old = cache.with_name(cache.stem + "_old")
        while old.with_suffix(".npy").exists():
            old = old.with_name(old.name + "+")
        os.replace(src=cache, dst=old.with_suffix(".npy"))
        if key_path.exists():
            os.replace(key_path, old.with_suffix(".json"))

    np.save(cache, img_flat)
    key_path.write_text(json.dumps(key, indent=2))
    custom_print(f"Saved cache ({img_flat.nbytes / 1e9:.2f} GB): {cache}")

    return subj_list, img_flat


def load_feat_table(
    tbl_path: str | Path,
    prefer_npz: bool = True,
    usecols: list[str] | None = None,
    id_col: str | None = None
) -> tuple[list[str], pd.DataFrame]:
    '''
    Load the feature matrix and the participant IDs which it belongs 
    from `tbl_path` (can be a CSV file or a NPZ file).
    
    If `prefer_npz` is set to True and the `tbl_path` points to a CSV file, 
    try to load from its sibling NPZ file (".npz" of the same stem).

    Although the layout of CSV / NPZ files differ slightly, 
    both should contain an ID column + named feature columns 
    (see also `make_df_*.py` scripts):
    - CSV: Participant IDs are stored along with the feature value, 
           with a header row storing their names.
           The ID column is named `id_col`, or the first column when `id_col` is omitted.
    - NPZ: Participant IDs are stored in the "SID" block, 
           feature values are stored in the "X" block, and
           feature names are stored in the "feats" block.

    One can use `usecols` to select features, 
    but the column order always follows the original table.

    Notes: 
    - If X is loaded from a NPZ file without a "feats" (p.s., cannot use `usecols`), 
      positional placeholders ("col-0", ...) will be generated.
    - Rows are dropped when the ID is missing or *any* selected feature is NaN.
    '''
    def _load_from_npz():
        with np.load(tbl_path, allow_pickle=True) as dat:
            if usecols is not None:
                assert "feats" in dat, "'usecols' needs the feature names, but the NPZ file has no 'feats' key"
                sel = np.isin(dat["feats"], usecols)  # boolean mask
                X = dat["X"][:, sel].astype(np.float32)
                feat_list = dat["feats"][sel].tolist()
                return (
                    dat["SID"], 
                    pd.DataFrame(X, columns=feat_list)
                )
            else:
                if "feats" not in dat:
                    custom_print(f"{tbl_path.name} has no 'feats' key; naming its columns positionally ('col-0', ...).")
                    feat_list = [ f"col-{i}" for i in range(dat["X"].shape[1]) ]
                else:
                    feat_list = dat["feats"].tolist()
                return (
                    dat["SID"], 
                    pd.DataFrame(dat["X"].astype(np.float32), columns=feat_list)
                )

    def _load_from_csv():
        nonlocal id_col
        df = pd.read_csv(tbl_path, usecols=usecols)
        id_col = df.columns.tolist()[0] if id_col is None else id_col
        feat_list = [ c for c in df.columns if c != id_col ]
        return (
            df[id_col].to_numpy(),  # convert into a np.array to use fancy indexing 
            df.loc[:, feat_list].astype(np.float32)
        )

    tbl_path = Path(tbl_path)

    if tbl_path.suffix.lower() not in [".npz", ".csv"]:
        raise ValueError(f"Unsupported table format: '{tbl_path.suffix}' ({tbl_path})")

    if prefer_npz:
        tbl_path = tbl_path.with_suffix(".npz")
    
    if tbl_path.suffix.lower() == ".npz" and tbl_path.exists():
        subj_arr, X = _load_from_npz()
    else:
        tbl_path = tbl_path.with_suffix(".csv")
        subj_arr, X = _load_from_csv()

    keep = X.notna().all(axis=1).to_numpy()
    subj_list = subj_arr[keep].tolist() 
    X = X[keep].reset_index(drop=True)

    custom_print(f"From: {tbl_path.name}")
    custom_print(f"{len(subj_list)} participants, {X.shape[1]} features")

    return subj_list, X


def get_latest_results(
    result_dir: Path = None, 
    search_level: str = "lv2", 
    return_level: str = "dir", 
    lv1_key: str = "*", 
    lv2_key: str = "*", 
    preds_fn: str | None = None, 
    timestamp_pattern: str = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
) -> Path:
    '''
    Find all second-level run or feature evaluation folders
    that match a specific pattern and contain prediction table(s).
    Than, return the one with the latest timestamp among the matches.
    '''
    if result_dir is None:
        proj_root = Path(__file__).resolve().parents[1]
        result_dir = proj_root / "results"

    if search_level == "lv2":
        preds_fn = preds_fn or "predictions.csv"
        pattern = os.path.join(lv1_key, lv2_key, preds_fn)
    elif search_level == "eval":
        preds_fn = preds_fn or "predictions_seed-*.csv"
        pattern = os.path.join(lv1_key, lv2_key, "eval_*", preds_fn)
    else:
        raise ValueError(f"`search_level` can only be 'lv2' or 'eval', got '{search_level}'")
    
    found = sorted({ p.parent for p in result_dir.glob(pattern) })
    assert found, f"\nNo {pattern} found under {result_dir}\n"
    latest_dir = max(found, key=lambda p: re.findall(timestamp_pattern, p.name))

    if return_level == "dir":
        return latest_dir
    elif return_level == "path":
        return latest_dir / preds_fn
    else:
        raise ValueError(f"`return_level` can only be 'dir' or 'path', got '{return_level}'")
    

def __getattr__(name: str):
    '''
    Models saved before `TargetScaler` moved to `models.py` were pickled
    under this module's name; resolve it lazily so they still load.
    '''
    if name in {"TargetScaler"}:
        import models
        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
