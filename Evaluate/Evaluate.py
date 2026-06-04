import json
import os
import random

import numpy as np
import pandas as pd
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


def evaluate(config):

    experiment_name = config["name"]
    print(f"current experiment:{experiment_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device:{device}")

    epochs = config["epochs"]
    start_epoch = 0
    print(f"total epochs:{epochs}")

    metric_average = config["metric_average"]
    pin_memory = True if torch.cuda.is_available() else False

    base_dir = os.path.join(config["save_dir"], experiment_name)
    os.makedirs(base_dir, exist_ok=True)
    best_save_path = os.path.join(base_dir, config["best_save_name"])
    current_save_path = os.path.join(base_dir, config["current_save_name"])
    result_dict_path = os.path.join(base_dir, config["result_dict_name"])

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
    encoder = get_encoder(config["encoder"]).to(device)
    decoder = get_decoder(config["decoder"]).to(device)
    optimizer = get_optimizer(
        config["optimizer"], list(encoder.parameters()) + list(decoder.parameters())
    )
    scheduler = get_scheduler(config["scheduler"], optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stop = EarlyStopping(mode="max", **config["early_stop"])
    scaler = GradScaler()

    result_dict = {
        "avg_train_loss": [],
        "avg_train_acc": [],
        "avg_val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
    }

    if os.path.exists(current_save_path):
        print("loading history models")
        current_checkpoint = torch.load(current_save_path)
        start_epoch = current_checkpoint["epoch"]
        encoder.load_state_dict(current_checkpoint["encoder"])
        decoder.load_state_dict(current_checkpoint["decoder"])
        optimizer.load_state_dict(current_checkpoint["optimizer"])
        scheduler.load_state_dict(current_checkpoint["scheduler"])
        early_stop.load_state_dict(current_checkpoint["early_stop"])
        scaler.load_state_dict(current_checkpoint["scaler"])
        optimizer.load_state_dict(current_checkpoint["optimizer"])

        if torch.cuda.is_available() and current_checkpoint["cuda_random"] is not None:
            torch.cuda.set_rng_state_all(current_checkpoint["cuda_random"])
        torch.random.set_rng_state(current_checkpoint["torch_random"])
        np.random.set_state(current_checkpoint["numpy_random"])
        random.setstate(current_checkpoint["python_random"])
        print("history model load successfully")

    if os.path.exists(result_dict_path):
        df = pd.read_csv(result_dict_path)
        result_dict = df.to_dict()

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        avg_train_loss, avg_train_acc = train(
            encoder,
            decoder,
            train_set_loader,
            train_itc_loader,
            optimizer,
            criterion,
            device,
            scaler,
        )
        print(
            f"[Train] [Epoch:{current_epoch}] avg_train_loss:{avg_train_loss},avg_train_acc:{avg_train_acc}"
        )
        result_dict["avg_train_acc"].append(avg_train_loss)
        result_dict["avg_train_acc"].append(avg_train_acc)

        avg_val_loss, val_acc, val_f1, val_auc = validate(
            encoder,
            decoder,
            val_set_loader,
            val_itc_loader,
            criterion,
            metric_average,
            device,
        )
        print(
            f"[Val] [Epoch:{current_epoch}] avg_val_loss:{avg_val_loss},val_acc:{val_acc},val_f1:{val_f1},val_auc:{val_auc}"
        )
        result_dict["avg_val_loss"].append(avg_val_loss)
        result_dict["val_acc"].append(val_acc)
        result_dict["val_f1"].append(val_f1)
        result_dict["val_auc"].append(val_auc)

        scheduler.step(avg_val_loss)
        is_improved = early_stop(val_f1)

        if not is_improved:
            print(
                f"[Early Stop] [Epoch:{current_epoch}],trigger counter: {early_stop.counter}/{early_stop.patience}"
            )
        else:
            early_stop.save_checkpoint(
                {
                    "best_epoch": current_epoch,
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                },
                best_save_path,
            )
            print(
                f"[Save Checkpoint] [Epoch:{current_epoch}],save best checkpoint successfully"
            )

        checkpoint = {
            "epoch": current_epoch,
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

        torch.save(checkpoint, current_save_path)
        print(f"[Save Checkpoint] [Epoch:{current_epoch}],save current checkpoint successfully")

        pd.DataFrame(result_dict).to_csv(result_dict_path, index=False)

        if early_stop.early_stop:
            print(f"[Save Checkpoint] [Epoch:{current_epoch}] trigger early_stop")
            break


def run_experiment(config_path):
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            evaluate(config=config)
    else:
        configs = os.listdir(config_path)
        for config in configs:
            current_path = os.path.join(config_path, config)
            with open(current_path, "r") as f:
                config = json.load(f)
                evaluate(config=config)
