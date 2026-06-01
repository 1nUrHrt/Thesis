import pandas as pd
from torch.utils.data import Subset, DataLoader
from Tools import DrugDataset

if __name__ == "__main__":
    node_path = "data/id2smiles.csv"
    itc_path = "data/KnownDDI-n.csv"
    train_durg_set_path = "./data/s1/train_set.csv"
    train_set = pd.read_csv(train_durg_set_path)
    drug_dataset = DrugDataset(node_path, False)
    train_drug_set = Subset(drug_dataset, train_set["drug_id"].tolist())

    print(train_drug_set)
