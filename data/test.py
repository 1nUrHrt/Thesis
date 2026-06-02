import json
import shutil
from matplotlib.offsetbox import DraggableAnnotation
import pandas as pd
from rdkit import Chem


import os
import pandas as pd
from rdkit import Chem

# # =========改这里两个路径=========
# sdf_folder = r"./data/DrugBank5.0_Approved_drugs"
# out_csv = r"./data/id2smiles.csv"
# id_list = []
# smiles_list = []

# for fname in os.listdir(sdf_folder):
#     if fname.endswith(".sdf"):
#         drug_id = fname.replace(".sdf", "")
#         sdf_path = os.path.join(sdf_folder, fname)

#         supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
#         mol = next(supplier) # 单个sdf只有一个分子，取第一个

#         if mol is None:
#             print(f"解析失败跳过：{fname}")
#             continue
#         mol_noH = Chem.RemoveHs(mol)
#         smi = Chem.MolToSmiles(mol_noH, isomericSmiles=True)
#         id_list.append(drug_id)
#         smiles_list.append(smi)

# df = pd.DataFrame({"id": id_list, "smiles": smiles_list})
# df.to_csv(out_csv, index=False)
# print(f"导出成功，共{len(df)}条，保存至：{out_csv}")


# itc = pd.read_csv(r"./data/KnownDDI.csv")
# print(itc.head())
# drugs = set(itc["Drug1"]).union(set(itc["Drug2"]))
# pd.Series(list(drugs)).to_csv(r"./data/drug_list.csv", index=False, header=["id"])

# drugs = pd.read_csv(r"./data/drug_list.csv")
# sdf_dir = r"./data/DrugBank5.0_Approved_drugs"
# save_dir = r"./data/drug_sdf"
# os.makedirs(save_dir, exist_ok=True)
# for drug_id in drugs['id']:
#     src = os.path.join(sdf_dir, f"{drug_id}.sdf")
#     dst = os.path.join(save_dir, f"{drug_id}.sdf")
#     shutil.copy(src, dst)

# dir = os.listdir(r"./data/drug_sdf")
# print(len(dir))

# base_dir = r"./data/ddi"
# split_dir = os.listdir(base_dir)
# df_list = []
# for outer in split_dir:
#     inner = os.listdir(os.path.join(base_dir, outer))
#     for file_name in inner:
#         file_path = os.path.join(base_dir, outer, file_name)
#         drug_df = pd.read_csv(file_path,sep=" ",names=["Drug1","Drug2","Label"])
#         df_list.append(drug_df)
# final_df = pd.concat(df_list, ignore_index=True)
# print(len(final_df))
# df = final_df.drop_duplicates(subset=["Drug1","Drug2","Label"], keep="first")
# df = df.sort_values(by="Label", ascending=True).reset_index(drop=True)
# print(df.head())
# df.to_csv(r"./data/KnownDDI-final.csv", index=False)
# itc = pd.read_csv(r"./data/KnownDDI-final.csv")

# with open("./data/node2id.json", "r", encoding="utf-8") as f:
#     dic = json.load(f)
# re_dic = {v: k for k, v in dic.items()}

# itc['Drug1'] = itc['Drug1'].map(re_dic)
# itc['Drug2'] = itc['Drug2'].map(re_dic)

# itc.to_csv(r"./data/KnownDDI-final-id.csv", index=False)
