import os.path

import pandas as pd

from sklearn.model_selection import train_test_split


def split_s1(base_dir, drug_file_name, itc_file_name, tr=0.7, va=0.1, te=0.2, seed=42):
    drug = pd.read_csv(str(os.path.join(base_dir, drug_file_name)))
    itc = pd.read_csv(str(os.path.join(base_dir, itc_file_name)))
    itc = itc.drop_duplicates(itc.columns, keep="first")

    train_df, temp_df = train_test_split(
        itc, test_size=1 - tr, random_state=seed, stratify=itc["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=te / (va + te), random_state=seed, stratify=temp_df["label"]
    )

    save_splits(
        os.path.join(base_dir, "s1"),
        collect_drugs(train_df),
        collect_drugs(val_df),
        collect_drugs(test_df),
        train_df,
        val_df,
        test_df,
    )


def collect_drugs(df):
    return (
        pd.concat([df["drug1"], df["drug2"]]).drop_duplicates().reset_index(drop=True)
    )


def save_splits(dir_path, train_drugs, val_drugs, test_drugs, train, val, test):

    header = ["id"]

    os.makedirs(dir_path, exist_ok=True)

    train_drugs.to_csv(
        os.path.join(dir_path, "train_set.csv"), index=False, header=header
    )
    val_drugs.to_csv(os.path.join(dir_path, "val_set.csv"), index=False, header=header)
    test_drugs.to_csv(
        os.path.join(dir_path, "test_set.csv"), index=False, header=header
    )

    train.to_csv(os.path.join(dir_path, "train.csv"), index=False)
    val.to_csv(os.path.join(dir_path, "val.csv"), index=False)
    test.to_csv(os.path.join(dir_path, "test.csv"), index=False)


if __name__ == "__main__":
    split_s1(
        base_dir="./data",
        drug_file_name="drug.csv",
        itc_file_name="KnownDDI.csv",
        tr=0.7,
        va=0.1,
        te=0.2,
    )
