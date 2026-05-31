import json
import os
import random
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler
from torch import nn
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from Tools import InteractionDataset, DrugDataset, wrapper_text, EarlyStopping
from Model import Encoder, Decoder


def train(encoder, decoder, drug_loader, train_loader, optimizer, criterion, scaler, device):
    if device == "cuda":
        AMP_DTYPE = torch.float16
    else:
        scaler = None
        AMP_DTYPE = torch.bfloat16

    encoder.train()
    decoder.train()

    train_loss = 0.0
    train_acc = 0.0

    for batch, (d1, d2, labels) in enumerate(train_loader):
        d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
        optimizer.zero_grad()
        all_drugs = []
        with autocast(device_type=device, dtype=AMP_DTYPE):
            for drugs in drug_loader:
                all_drugs.append(encoder(drugs.to(device)))
            all_drugs = torch.cat(all_drugs, dim=0)
            logits = decoder(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)

        acc = (preds == labels).float().mean()

        train_loss += loss.item()
        train_acc += acc.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    return avg_train_loss, avg_train_acc


def validation(encoder, decoder, drug_loader, val_loader, criterion, metric, device):
    encoder.eval()
    decoder.eval()

    val_loss = 0.0
    val_acc = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        all_drugs = []
        for drugs in drug_loader:
            all_drugs.append(encoder(drugs.to(device)))
        all_drugs = torch.cat(all_drugs, dim=0)

        for batch, (d1, d2, labels) in enumerate(val_loader):
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = decoder(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            acc = (preds == labels).float().mean()

            val_loss += loss.item()
            val_acc += acc.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(
        all_labels,
        all_preds,
        average='macro'
    )
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    return avg_val_loss, avg_val_acc, val_f1


def run_experiment(config_path):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    for config in configs:
        evaluate(config)


# ─── Encoder / Decoder registry ───────────────────────────────────────────
# Maps config type string → (module_name, class_name)
ENCODER_REGISTRY = {
    "mine": ("Model", "Encoder"),
    "gin": ("Baseline", "GINEncoder"),
    "gcn": ("Baseline", "GCNEncoder"),
    "identity": ("Baseline", "Encoder"),
}

DECODER_REGISTRY = {
    "mine": ("Model", "Decoder"),
    "identity": ("Baseline", "Decoder"),
}


def _import_class(module_name: str, class_name: str):
    mod = __import__(module_name, fromlist=[class_name])
    return getattr(mod, class_name)


def create_encoder(config):
    mod_name, cls_name = ENCODER_REGISTRY[config["type"]]
    cls = _import_class(mod_name, cls_name)
    return cls(**config['params'])


def create_decoder(config):
    mod_name, cls_name = DECODER_REGISTRY[config["type"]]
    cls = _import_class(mod_name, cls_name)
    return cls(**config['params'])


def create_optimizer(config, model_parameters):
    optimizer_cls = getattr(torch.optim, config['type'])
    optimizer = optimizer_cls(model_parameters, **config['params'])
    return optimizer


def create_scheduler(config, optimizer):
    scheduler_cls = getattr(torch.optim.lr_scheduler, config["type"])
    scheduler = scheduler_cls(optimizer, **config['params'])
    return scheduler


def evaluate(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device:{device}")

    pin_memory = True if torch.cuda.is_available() else False

    drug_dataset = DrugDataset(**config['drug_dataset']['params'])
    itc_dataset = InteractionDataset(**config['itc_dataset']['params'])
    encoder = create_encoder(config["encoder"])
    decoder = create_decoder(config["decoder"])
    optimizer = create_optimizer(config["optimizer"], list(encoder.parameters()) + list(decoder.parameters()))
    scheduler = create_scheduler(config['scheduler'], optimizer)
    criterion = nn.CrossEntropyLoss(**config['criterion'])
    early_stop = EarlyStopping(**config["early_stop"])
    return encoder, decoder, optimizer, scheduler, criterion, early_stop


run_experiment("./config.json")
