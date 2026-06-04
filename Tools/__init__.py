from .InteractionDataset import InteractionDataset
from .DrugDataset import DrugDataset
from .EarlyStopping import EarlyStopping


import os
from typing import Literal
from .SubDrugDataset import SubDrugDataset
from AttnEncoder import AttnEncoder
from AttnResEncoder import AttnResEncoder
from GeneralModel import DefaultDecoder
import torch


def get_encoder(config):
    model_name = config["type"]
    kwargs = config["params"]
    if model_name == "AttnEncoder":
        return AttnEncoder(**kwargs)
    elif model_name == "AttnResEncoder":
        return AttnResEncoder(**kwargs)
    else:
        raise NotImplementedError("Model {} not implemented.".format(model_name))


def get_decoder(config):
    model_name = config["type"]
    kwargs = config["params"]
    if model_name == "DefaultDecoder":
        return DefaultDecoder(**kwargs)
    else:
        raise NotImplementedError("Model {} not implemented.".format(model_name))


def get_optimizer(config, model_parameters):
    optimizer_cls = getattr(torch.optim, config["type"])
    optimizer = optimizer_cls(model_parameters, **config["params"])
    return optimizer


def get_scheduler(config, optimizer):
    scheduler_cls = getattr(torch.optim.lr_scheduler, config["type"])
    scheduler = scheduler_cls(optimizer, **config["params"])
    return scheduler


def load_dataset(base_dir: str, split_type: Literal["s1", "s2", "s3"]):

    def get_path(file_name):
        return os.path.join(base_dir, split_type, file_name)

    drug_set = DrugDataset(os.path.join(base_dir, "drug.csv"))

    train_set = SubDrugDataset(get_path("train_set.csv"), drug_set)
    val_set = SubDrugDataset(get_path("val_set.csv"), drug_set)
    test_set = SubDrugDataset(get_path("test_set.csv"), drug_set)
    train_itc = InteractionDataset(get_path("train.csv"))
    val_itc = InteractionDataset(get_path("val.csv"))
    test_itc = InteractionDataset(get_path("test.csv"))
    return train_set, train_itc, val_set, val_itc, test_set, test_itc


# COLORS = {
#     "info": "\033[92m",  # 绿色
#     "debug": "\033[94m",  # 蓝色
#     "warning": "\033[93m",  # 黄色
#     "error": "\033[91m",  # 红色
#     "reset": "\033[0m",
# }


# def wrapper_text(text, mode):
#     return f"{COLORS[mode]}{text}{COLORS['reset']}"


__all__ = [
    "InteractionDataset",
    "DrugDataset",
    "EarlyStopping",
    "get_encoder",
    "get_decoder",
    "get_optimizer",
    "get_scheduler",
    "load_dataset",
    "SubDrugDataset",
]
