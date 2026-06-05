import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_ddi_by_drug_scenario(
    base_dir: str,
    input_file: str,
    output_dir: str = "split_scenarios",
    seen_drug_ratio: float = 0.7,
    train_ratio_of_s0: float = 0.7,
    val_ratio_of_rest: float = 0.5,
    random_seed: int = 42,
    stratify_by_label: bool = True,
    label_col: str = "label",
    drug1_col: str = "drug1",
    drug2_col: str = "drug2",
):
    np.random.seed(random_seed)

    # 1. 读取数据
    df = pd.read_csv(os.path.join(base_dir, input_file))
    df = df.drop_duplicates(
        subset=[drug1_col, drug2_col, label_col], keep="first"
    ).reset_index(drop=True)
    print(f"原始数据大小: {len(df)}")

    # 2. 获取所有药物并划分 seen / unseen
    all_drugs = pd.concat([df[drug1_col], df[drug2_col]]).unique()
    n_seen = int(len(all_drugs) * seen_drug_ratio)
    seen_drugs = set(np.random.choice(all_drugs, size=n_seen, replace=False))
    unseen_drugs = set(all_drugs) - seen_drugs
    print(f"Seen 药物数: {len(seen_drugs)}, Unseen 药物数: {len(unseen_drugs)}")

    # 3. 标记每行的场景类型 (S0, S1, S2)
    def get_scenario(row):
        d1_seen = row[drug1_col] in seen_drugs
        d2_seen = row[drug2_col] in seen_drugs
        if d1_seen and d2_seen:
            return "S0"
        elif d1_seen or d2_seen:
            return "S2"
        else:
            return "S1"

    df["scenario"] = df.apply(get_scenario, axis=1)

    # 4. 分离 S0 样本
    s0_all = df[df["scenario"] == "S0"].copy()
    print(f"S0 样本数: {len(s0_all)}")

    # label_counter = Counter(s0_all["label"])
    # label_count_df = pd.DataFrame(label_counter.items(), columns=["label", "count"])
    # print(label_count_df)
    if len(s0_all) == 0:
        raise ValueError(
            "没有 S0 样本，请减小 seen_drug_ratio 或检查数据中药物出现的广度。"
        )

    # 5. 从 S0 中划分训练集和剩余 S0
    if stratify_by_label and len(s0_all[label_col].unique()) > 1:
        stratify = s0_all[label_col]
    else:
        stratify = None

    train_s0, rest_s0 = train_test_split(
        s0_all,
        train_size=train_ratio_of_s0,
        random_state=random_seed,
        stratify=stratify,
    )
    train_set = pd.concat([train_s0[drug1_col], train_s0[drug2_col]]).drop_duplicates()
    train_map_dict = {k: i for i, k in enumerate(train_set)}
    train_s0[drug1_col] = train_s0[drug1_col].map(train_map_dict)
    train_s0[drug2_col] = train_s0[drug2_col].map(train_map_dict)

    # 6. 收集剩余样本（rest_s0 + 全部 S2 + 全部 S1）
    s2_all = df[df["scenario"] == "S2"].copy()
    s1_all = df[df["scenario"] == "S1"].copy()

    rest_df = pd.concat([rest_s0, s2_all, s1_all], ignore_index=True)

    # 7. 按场景类型分层划分验证集和测试集
    if stratify_by_label:
        # 使用场景类型作为分层依据
        stratify_rest = rest_df["scenario"]
    else:
        stratify_rest = None

    val_df, test_df = train_test_split(
        rest_df,
        test_size=1 - val_ratio_of_rest,
        random_state=random_seed,
        stratify=stratify_rest,
    )
    val_set = pd.concat([val_df[drug1_col], val_df[drug2_col]]).drop_duplicates()
    val_map_dict = {k: i for i, k in enumerate(val_set)}
    val_df[drug1_col] = val_df[drug1_col].map(val_map_dict)
    val_df[drug2_col] = val_df[drug2_col].map(val_map_dict)

    test_set = pd.concat([test_df[drug1_col], test_df[drug2_col]]).drop_duplicates()
    test_map_dict = {k: i for i, k in enumerate(test_set)}
    test_df[drug1_col] = test_df[drug1_col].map(test_map_dict)
    test_df[drug2_col] = test_df[drug2_col].map(test_map_dict)

    # 9. 保存结果
    os.makedirs(output_dir, exist_ok=True)

    train_set.to_csv(
        os.path.join(output_dir, "train_set.csv"), index=False, header=["id"]
    )
    
    val_set.to_csv(
        os.path.join(output_dir, "val_set.csv"), index=False, header=["id"]
    )
    
    test_set.to_csv(
        os.path.join(output_dir, "test_set.csv"), index=False, header=["id"]
    )


    train_s0.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # 保存 seen/unseen 列表
    pd.Series(list(seen_drugs)).to_csv(
        os.path.join(output_dir, "seen_drugs.csv"), index=False, header=["id"]
    )
    pd.Series(list(unseen_drugs)).to_csv(
        os.path.join(output_dir, "unseen_drugs.csv"), index=False, header=["id"]
    )

    # 10. 输出统计报告
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DDI 多分类数据集划分报告")
    report_lines.append("=" * 60)
    report_lines.append(
        f"总药物数: {len(all_drugs)} | Seen: {len(seen_drugs)} | Unseen: {len(unseen_drugs)}"
    )
    report_lines.append(f"总样本数: {len(df)}")
    report_lines.append(f"训练集 (仅 S0): {len(train_s0)}")
    report_lines.append(f"验证集: {len(val_df)} | 测试集: {len(test_df)}")
    report_lines.append("\n--- 各集合场景分布 ---")
    for name, dset in [("训练集", train_s0), ("验证集", val_df), ("测试集", test_df)]:
        cnt = dset["scenario"].value_counts().to_dict()
        report_lines.append(f"{name}: {cnt}")
    report_lines.append("\n--- 各集合标签分布（前5类）---")
    for name, dset in [("训练集", train_s0), ("验证集", val_df), ("测试集", test_df)]:
        label_cnt = dset[label_col].value_counts().head(5).to_dict()
        report_lines.append(f"{name} label 分布: {label_cnt}")

    with open(os.path.join(output_dir, "split_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    # 打印报告
    print("\n".join(report_lines))



if __name__ == "__main__":
   
    split_ddi_by_drug_scenario(
        base_dir="./data",
        input_file="KnownDDI.csv",
        output_dir="./data/split_s0_s1_s2",
        seen_drug_ratio=0.9,
        train_ratio_of_s0=0.8,  
        val_ratio_of_rest=0.5,  
        random_seed=45,
        stratify_by_label=True,
        label_col="label",
        drug1_col="drug1",
        drug2_col="drug2",
    )

