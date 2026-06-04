import torch
from torch.utils.data import DataLoader
from Tools import DrugDataset, InteractionDataset
from torch_geometric.data import Batch
from AttnEncoder import AttnEncoder
import pandas as pd

if __name__ == "__main__":
    # node_path = "data/s1/val_and_test_set.csv"
    # itc_path = "data/s1/test.csv"
    # drug_dataset = DrugDataset(node_path)
    # itc_dataset = InteractionDataset(itc_path)
    # drug_loader = DataLoader(
    #     drug_dataset,
    #     batch_size=1024,
    #     shuffle=False,
    #     collate_fn=lambda batch: Batch.from_data_list(batch),
    # )
    # itc_loader = DataLoader(
    #     itc_dataset,
    #     batch_size=1024,
    #     shuffle=True,
    #     collate_fn=itc_dataset.itc_collate_fn,
    # )
    # encoder = AttnEncoder(39, 10, 128, 1, 0.1, 8)
    # drug_arr = []
    # for batch in drug_loader:
    #     drugs = encoder(batch)
    #     drug_arr.append(drugs)
    # x = torch.cat(drug_arr, dim=0)
    # for batch in itc_loader:
    #     drug1,drug2,label = batch
    #     print(x[drug1])

    node_path = "data/s1/val_and_test_set.csv"
    itc_path = "data/s1/test.csv"
    drug_dataset = DrugDataset(node_path)
    itc_dataset = InteractionDataset(itc_path)
    drug_loader = DataLoader(
        drug_dataset,
        batch_size=1024,
        shuffle=False,
        collate_fn=lambda batch: Batch.from_data_list(batch),
    )
    itc_loader = DataLoader(
        itc_dataset,
        batch_size=1024 * 10 * 2,
        shuffle=True,
        collate_fn=itc_dataset.itc_collate_fn,
    )
    print(len(drug_dataset))
    for batch in itc_loader:
        drug1, drug2, label = batch
        drugs = torch.cat([drug1,drug2]).unique()
        print(drugs.shape)
