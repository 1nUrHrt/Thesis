import os.path

import pandas as pd

from sklearn.model_selection import train_test_split


def split_s1(base_dir, file_name, tr=0.7, va=0.1, te=0.2, seed=42):
    df = pd.read_csv(str(os.path.join(base_dir, file_name)))
    df = df.drop_duplicates(df.columns, keep="first")

    train_df, temp_df = train_test_split(
        df, test_size=1 - tr, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=te / (va + te), random_state=seed, stratify=temp_df["label"]
    )
    save_splits(os.path.join(base_dir, "s1"), train_df, val_df, test_df)


def collect_drugs(df):
    drugs = set(df["drug1"]).union(set(df["drug2"]))
    return sorted(list(drugs))


def save_splits(dir_path, train, val, test):
    train_drugs = collect_drugs(train)
    val_drugs = collect_drugs(val)
    test_drugs = collect_drugs(test)

    os.makedirs(dir_path, exist_ok=True)

    pd.Series(train_drugs).to_csv(os.path.join(dir_path, "train_set.csv"), index=False, header=["drug_id"])
    pd.Series(val_drugs).to_csv(os.path.join(dir_path, "val_set.csv"), index=False, header=["drug_id"])
    pd.Series(test_drugs).to_csv(os.path.join(dir_path, "test_set.csv"), index=False, header=["drug_id"])

    train.to_csv(os.path.join(dir_path, "train.csv"), index=False)
    val.to_csv(os.path.join(dir_path, "val.csv"), index=False)
    test.to_csv(os.path.join(dir_path, "test.csv"), index=False)


if __name__ == "__main__":
    split_s1(base_dir="../data", file_name="KnownDDI.csv", tr=0.7, va=0.1, te=0.2)
