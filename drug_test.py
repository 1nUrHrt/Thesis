import pandas as pd
from torch.utils.data import Subset, DataLoader
from Tools import DrugDataset

if __name__ == "__main__":
    node_path = "data/s1/test_set.csv"
    itc_path = "data/s1/test.csv"
    drug_dataset = DrugDataset(node_path)
    # itc_df = pd.read_csv(itc_path)

    print(drug_dataset[0])
