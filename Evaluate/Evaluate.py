import json

import torch
from sklearn.metrics import f1_score
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
    train_loader,
    optimizer,
    criterion,
    device,
    scaler=None,
):
    if device == "cuda":
        AMP_DTYPE = torch.float16
    else:
        scaler = None
        AMP_DTYPE = torch.bfloat16

    encoder.train()
    decoder.train()

    train_loss = 0.0
    train_acc = 0.0

    for d1, d2, labels in train_loader:
        d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device, dtype=AMP_DTYPE):
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

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    return avg_train_loss, avg_train_acc


def validation(encoder, decoder, drug_loader, val_loader, criterion, average, device):
    encoder.eval()
    decoder.eval()

    val_loss = 0.0
    val_acc = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in val_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = decoder(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            acc = (preds == labels).float().mean()

            val_loss += loss.item()
            val_acc += acc.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average=average)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    return avg_val_loss, avg_val_acc, val_f1


def run_experiment(config_path):
    with open(config_path, "r") as f:
        configs = json.load(f)
    for config in configs:
        evaluate(config)


def evaluate(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device:{device}")

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

    train(
        encoder,
        decoder,
        train_set_loader,
        train_itc_loader,
        optimizer,
        criterion,
        scaler,
        device,
    )


run_experiment("./config.json")
