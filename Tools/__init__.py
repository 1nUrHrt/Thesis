from .InteractionDataset import InteractionDataset
from .DrugDataset import DrugDataset
from .EarlyStopping import EarlyStopping


import os
from typing import Literal
from .SubDrugDataset import SubDrugDataset
import Encoder
import Classifier
import torch

import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def get_encoder(config):
    model_name = config["type"]
    if model_name not in Encoder.__all__:
        raise NotImplementedError("Enocoder {} not supported.".format(model_name))
    return getattr(Encoder, model_name)(**config["params"])


def get_classifier(config):
    model_name = config["type"]
    if model_name not in Classifier.__all__:
        raise NotImplementedError("Classifier {} not supported.".format(model_name))
    return getattr(Classifier, model_name)(**config["params"])


def get_optimizer(config, model_parameters):
    optimizer_cls = getattr(torch.optim, config["type"])
    optimizer = optimizer_cls(model_parameters, **config["params"])
    return optimizer


def get_scheduler(config, mode, optimizer):
    scheduler_cls = getattr(torch.optim.lr_scheduler, config["type"])
    scheduler = scheduler_cls(optimizer, mode=mode, **config["params"])
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
    "get_classifier",
    "get_optimizer",
    "get_scheduler",
    "load_dataset",
    "SubDrugDataset",
    "Timer",
]
