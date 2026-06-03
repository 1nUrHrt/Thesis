from typing import Literal
from Tools import DrugDataset, InteractionDataset
import os



def load_dataset(base_dir:str,split_type: Literal["s1", "s2", "s3"]):

    def get_path(file_name):
        return os.path.join(base_dir,split_type,file_name)

    os.path.join(base_dir,split_type,)
    train_set = DrugDataset(get_path("train_set.csv"))
    train_itc = InteractionDataset(get_path("train.csv"))
    val_and_test_set = DrugDataset(get_path("val_and_test_set.csv"))
    val_itc = InteractionDataset(get_path("val.csv"))
    test_itc = InteractionDataset(get_path("test.csv"))
    return train_set, train_itc, val_and_test_set, val_itc, test_itc