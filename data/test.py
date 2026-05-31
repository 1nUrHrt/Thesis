import json
import pandas as pd
from rdkit import Chem


def get_all_atoms():
    with open("./id2smiles.json", "r") as f:
        id2smiles = json.load(f)
    atom_count = {}
    for i in id2smiles:
        mol = Chem.MolFromSmiles(id2smiles[i])
        if mol is None:
            raise ValueError("Invalid SMILES")
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in atom_count:
                atom_count[atom_symbol] = 1
            else:
                atom_count[atom_symbol] += 1
    print(atom_count)


if __name__ == '__main__':
    # df = pd.read_csv("./KnownDDI.csv")
    # df.columns = ["drug1", "drug2", "label"]
    # df = df[df["label"] < 40]
    # df.to_csv("./KnownDDI-p.csv", index=False)
    x = 10
    y = 2
    print(x // 3)
