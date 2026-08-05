#!/usr/bin/env python3

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils import get_col_names


class Config:
    def __init__(self):
        self.y_col = "BASIC_INFO_AGE"
        self.impute_strategy = "median"
        self.setup_paths()

    def setup_paths(self):
        proj_root = Path(__file__).resolve().parents[1]
        self.tbl_merged_path            = proj_root / "data" / "tabular" / "df_merged.csv"
        self.tbl_main_path              = proj_root / "data" / "tabular" / "DATA_ses-01_2025-05-29.csv"
        self.tbl_basic_path             = proj_root / "data" / "tabular" / "Questionnaires_ses-01_number-coding.csv"
        self.tbl_str_path               = proj_root / "data" / "tabular" / "STRUCTURE_VOLUME_MRI_ses-01_2025-05-23.csv"
        self.tbl_motor_path             = proj_root / "data" / "tabular" / "motor_summary_ses-01_new.csv"
        self.tbl_language_path          = proj_root / "data" / "tabular" / "language_summary_ses01.csv"
        self.traintest_split_path       = proj_root / "data" / "meta" / "train-test_subjs.json"
        self.dropped_feature_names_path = proj_root / "data" / "meta" / "preproc_features_dropped.txt"
        self.feature_names_path         = proj_root / "data" / "meta" / "preproc_features.txt"
        self.tbl_out_path               = proj_root / "data" / "tabular" / "df_preproc.csv"


def load_data(paths: dict[str, Path]) -> pd.DataFrame:
    '''
    Load and merge multiple CSV files into a single DataFrame.
    '''
    df = pd.read_csv(paths["main"], index_col=0)

    df_basic = pd.read_csv(paths["basic"], index_col=0)
    df_basic = df_basic.rename(columns={ c: f"BASIC_INFO_{c}" for c in get_col_names("basic_info_cols")) })
    df_basic = df_basic.rename(columns={ c: f"BASIC_Q_{c}" for c in get_col_names("basic_q_cols") })
    df.update(df_basic)

    df_str = pd.read_csv(paths["structure"], index_col=0)
    df_str = df_str.rename(columns={ c: f"STRUCTURE_{c}" for c in get_col_names("structure_cols") })
    df.update(df_str)

    df_motor = pd.read_csv(paths["motor"], index_col=0)
    df_motor = df_motor.rename(columns={ c: f"MOTOR_{c}" for c in get_col_names("motor_cols") })
    df.update(df_motor)

    df_language = pd.read_csv(paths["language"], index_col=0)
    df_language = df_language.rename(columns={ c: f"LANGUAGE_{c}" for c in get_col_names("language_cols") })
    df.update(df_language)

    return df


def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[
        c for c in df.columns if 
        ( c.startswith("BASIC_") and (c not in ["BASIC_INFO_AGE", "BASIC_INFO_SEX"]) )
        or ( "_ST_" in c )
    ])


def split_data(traintest_split_path: Path, df: pd.DataFrame):
    with open(traintest_split_path, "r") as f:
        split_idxs = json.load(f)

    return (
        df.loc[split_idxs["train"], :], 
        df.loc[split_idxs["test"], :]
    )


def remove_outlier_features(df_train, df_test):
    na_rates = pd.Series(df_train.isnull().sum() / len(df_train))
    q1 = na_rates.quantile(.25)
    q3 = na_rates.quantile(.75)
    iqr = q3 - q1
    outliers = na_rates[na_rates > (q3 + iqr*1.5)]

    return (
        df_train.drop(columns=outliers.index), 
        df_test.drop(columns=outliers.index), 
        outliers.index
    )


def fill_missing(df_train, df_test, features, impute_strategy):
    imputer = SimpleImputer(strategy=impute_strategy)
    df_train.loc[:, features] = imputer.fit_transform(df_train.loc[:, features])
    df_test.loc[:, features] = imputer.transform(df_test.loc[:, features])
    
    return df_train, df_test


def scale_features(df_train, df_test, features):
    scaler = StandardScaler()
    df_train.loc[:, features] = scaler.fit_transform(df_train.loc[:, features])
    df_test.loc[:, features] = scaler.transform(df_test.loc[:, features])
    
    return df_train, df_test


def main():
    config = Config()

    if config.tbl_merged_path.exists():
        data = pd.read_csv(config.tbl_merged_path, index_col=0)
    else:
        data = load_data({
            "main": config.tbl_main_path, 
            "basic": config.tbl_basic_path, 
            "structure": config.tbl_str_path,
            "motor": config.tbl_motor_path, 
            "language": config.tbl_language_path
        })
        data.to_csv(config.tbl_merged_path)

    assert config.y_col in data.columns, f"Target column '{config.y_col}' not in the dataset."
    
    data = drop_cols(data)
    df_train, df_test = split_data(config.traintest_split_path, data)

    df_train, df_test, outlier_features = remove_outlier_features(df_train, df_test)
    features = [ c for c in df_train.columns if c not in ["BASIC_INFO_AGE", "BASIC_INFO_SEX"] ]
    with open(config.dropped_feature_names_path, "w") as f:
        f.write("\n".join(outlier_features))
    with open(config.feature_names_path, "w") as f:
        f.write("\n".join(features))

    df_train, df_test = fill_missing(df_train, df_test, features, config.impute_strategy)
    df_train, df_test = scale_features(df_train, df_test, features)
    
    df_train.insert(0, "Set", "train")
    df_test.insert(0, "Set", "test")
    config.tbl_out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df_train, df_test], axis=0).to_csv(config.tbl_out_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
    print("\nFinish data preprocessing.\n")