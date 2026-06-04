import json
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch import nn
from torch.utils.data import DataLoader
from Tools import EarlyStopping
from Tools import get_encoder, get_decoder, get_optimizer, get_scheduler, load_dataset


def train(
    encoder,
    decoder,
    drug_loader,
    itc_loader,
    optimizer,
    criterion,
    device,
    scaler=None,
):

    encoder.train()
    decoder.train()

    train_loss = 0.0
    train_acc = 0.0

    for d1, d2, labels in itc_loader:
        d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device):
            all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])
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

    avg_train_loss = train_loss / len(itc_loader)
    avg_train_acc = train_acc / len(itc_loader)
    return avg_train_loss, avg_train_acc


def validate(
    encoder, decoder, drug_loader, itc_loader, criterion, metric_average, device
):
    encoder.eval()
    decoder.eval()

    val_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in itc_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = decoder(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            val_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    avg_val_loss = val_loss / len(itc_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average=metric_average, zero_division=0)
    val_auc = roc_auc_score(
        all_labels, all_probs, multi_class="ovr", average=metric_average
    )
    return avg_val_loss, val_acc, val_f1, val_auc


def test(encoder, decoder, drug_loader, itc_loader, criterion, metric_average, device):
    encoder.eval()
    decoder.eval()

    val_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in itc_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = decoder(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            val_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(prob.cpu().numpy())

    avg_val_loss = val_loss / len(itc_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average=metric_average, zero_division=0)
    val_auc = roc_auc_score(
        all_labels, all_probs, multi_class="ovr", average=metric_average
    )
    return avg_val_loss, val_acc, val_f1, val_auc


def run_experiment(config_path):
    with open(config_path, "r") as f:
        configs = json.load(f)
    for config in configs:
        evaluate(config)


def evaluate(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device:{device}")

    epochs = config["epochs"]
    metric_average = config["metric_average"]
    pin_memory = True if torch.cuda.is_available() else False

    train_set, train_itc, val_set, val_itc, test_set, test_itc = load_dataset(
        **config["dataset"]
    )
    loader_config = config["data_loader"]
    train_set_loader = DataLoader(
        train_set,
        collate_fn=train_set.drug_collate_fn,
        pin_memory=pin_memory,
        **loader_config["train_set"],
    )
    val_set_loader = DataLoader(
        train_set,
        collate_fn=val_set.drug_collate_fn,
        pin_memory=pin_memory,
        **loader_config["val_set"],
    )
    test_set_loader = DataLoader(
        train_set,
        collate_fn=test_set.drug_collate_fn,
        pin_memory=pin_memory,
        **loader_config["test_set"],
    )
    train_itc_loader = DataLoader(
        train_itc,
        collate_fn=train_itc.itc_collate_fn,
        pin_memory=pin_memory,
        **loader_config["train_itc"],
    )
    val_itc_loader = DataLoader(
        val_itc,
        collate_fn=val_itc.itc_collate_fn,
        pin_memory=pin_memory,
        **loader_config["val_itc"],
    )
    test_itc_loader = DataLoader(
        test_itc,
        collate_fn=test_itc.itc_collate_fn,
        pin_memory=pin_memory,
        **loader_config["test_itc"],
    )
    encoder = get_encoder(config["encoder"])
    decoder = get_decoder(config["decoder"])
    optimizer = get_optimizer(
        config["optimizer"], list(encoder.parameters()) + list(decoder.parameters())
    )
    scheduler = get_scheduler(config["scheduler"], optimizer)
    criterion = nn.CrossEntropyLoss(**config["criterion"])
    early_stop = EarlyStopping(**config["early_stop"])
    scaler = GradScaler()
    for epoch in range(epochs):
        avg_train_loss, avg_train_acc = train(
            encoder,
            decoder,
            train_set_loader,
            train_itc_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        avg_val_loss, val_acc, val_f1, val_auc = validate(
            encoder,
            decoder,
            val_set_loader,
            val_itc_loader,
            criterion,
            metric_average,
            device,
        )
        scheduler.step(avg_val_loss)
        is_improved = early_stop(
            val_f1,
            epoch + 1,
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            },
        )

        checkpoint = {
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stop": early_stop.state_dict(),
            "scaler": scaler.state_dict(),
            "cuda_random": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "torch_random": torch.random.get_rng_state(),
            "numpy_random": np.random.get_state(),
            "python_random": random.getstate(),
        }

        if not is_improved:
            print(f"trigger counter: {early_stop.counter}/{early_stop.patience}")

        if early_stop.early_stop:
            print("trigger early_stop")


run_experiment("./config.json")
