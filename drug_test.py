from Tools import DrugDataset, InteractionDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    node_path = 'data/id2smiles.csv'
    itc_path = "data/KnownDDI-n.csv"
    drug_dataset = DrugDataset(node_path, True)
    itc_dataset = InteractionDataset(itc_path)
    itc_loader = DataLoader(itc_dataset, batch_size=32, shuffle=False, collate_fn=itc_dataset.itc_collate_fn)
    print(itc_dataset.labels)
