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

if __name__ == '__main__':
    node_path = 'data/id2smiles.csv'
    itc_path = "data/KnownDDI.csv"
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, "best.pt")
    current_checkpoint_path = os.path.join(save_dir, "current.pt")
    record_path = os.path.join(save_dir, "record.csv")

    epoch_bar_format = wrapper_text(f"{{desc}} Elapsed:{{elapsed}}{{postfix}}", "info")

    drug_dataset = DrugDataset(node_path, False)
    itc_data = InteractionDataset(itc_path=itc_path)

    node_dim, edge_dim = drug_dataset.get_size()
    h_dim = 128
    layer_num = 5
    heads = 8
    dp_r = 0.2

    drug_batch_size = 1024 * 2
    itc_batch_size = 1024 * 10 * 2

    start_epoch = -1
    epochs = 200

    lr = 1e-3
    label_smoothing = 0.1
    weight_decay = 1e-5

    class_num = 87
    val_split = 0.1
    patience = 10
    early_stop_delta = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if torch.cuda.is_available() else False

    print(f"device:{device}")

    indices = np.arange(len(itc_data))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=42,
        stratify=itc_data.labels
    )

    train_dataset = Subset(itc_data, train_idx)
    val_dataset = Subset(itc_data, val_idx)

    drug_loader = DataLoader(drug_dataset, batch_size=drug_batch_size, shuffle=False, num_workers=0,
                             pin_memory=pin_memory,
                             collate_fn=drug_dataset.drug_collate_fn)

    train_loader = DataLoader(
        train_dataset,
        batch_size=itc_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=itc_data.itc_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=itc_batch_size * 2,
        shuffle=False,
        num_workers=2,
        collate_fn=itc_data.itc_collate_fn
    )

    encoder = Encoder(
        node_dim,
        edge_dim,
        h_dim=h_dim,
        block_num=layer_num,
        dp_r=dp_r,
        heads=heads,

    ).to(device)

    decoder = Decoder(
        in_feature=h_dim,
        out_feature=class_num,
        dp_r=dp_r,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    if device == "cuda":
        scaler = GradScaler("cuda")
        AMP_DTYPE = torch.float16
    else:
        scaler = None
        AMP_DTYPE = torch.bfloat16

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    early_stop = EarlyStopping(
        save_path=best_checkpoint_path,
        patience=patience,
        min_delta=early_stop_delta,
        mode="max"
    )

    history_record = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    if os.path.exists(current_checkpoint_path):
        print("loading checkpoint...")
        checkpoint = torch.load(current_checkpoint_path, weights_only=False)
        start_epoch = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint["scaler"])
        early_stop.load_state_dict(checkpoint['early_stop'])
        cuda_state = checkpoint["cuda_random"]
        if isinstance(cuda_state, list):
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            torch.cuda.set_rng_state(cuda_state)
        torch.random.set_rng_state(checkpoint["torch_random"])
        np.random.set_state(checkpoint["numpy_random"])
        random.setstate(checkpoint["python_random"])
        del checkpoint
        print("checkpoint loaded")

    if os.path.exists(record_path):
        print("loading statistics...")
        df = pd.read_csv(record_path)
        for key in history_record.keys():
            if key in df.columns:
                history_record[key] = df[key].tolist()
        del df
        print("statistics loaded")

    print("start training....")

    epoch_pbar = tqdm(
        range(start_epoch + 1, epochs),
        bar_format=epoch_bar_format,
    )

    for epoch in epoch_pbar:

        encoder.train()
        decoder.train()

        train_loss = 0.0
        train_acc = 0.0

        for batch, (d1, d2, labels) in enumerate(train_loader):
            epoch_pbar.set_description_str(
                f"Epoch:[{epoch + 1}/{epochs}] [Train] Batch:[{batch + 1}/{len(train_loader)}]", refresh=True)
            d1 = d1.to(device)
            d2 = d2.to(device)
            labels = labels.to(device)
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

            epoch_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc.item():.4f}"
            }, refresh=True)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        history_record['train_loss'].append(avg_train_loss)
        history_record['train_acc'].append(avg_train_acc)

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
                epoch_pbar.set_description_str(
                    f"Epoch:[{epoch + 1}/{epochs}] [Val] Batch:[{batch + 1}/{len(val_loader)}]", refresh=True)
                d1 = d1.to(device)
                d2 = d2.to(device)
                labels = labels.to(device)
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

        history_record['val_loss'].append(avg_val_loss)
        history_record['val_acc'].append(avg_val_acc)
        history_record['val_f1'].append(val_f1)

        epoch_pbar.write(
            wrapper_text(
                f"Epoch:[{epoch + 1}/{epochs}] Val Loss:{avg_val_loss:.4f},Val Acc:{val_acc:.4f},Val F1:{val_f1:.4f}",
                "debug")
        )

        # scheduler
        scheduler.step(avg_val_loss)

        is_increase = early_stop(val_f1, {
            'best_epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
        })

        checkpoint = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stop": early_stop.state_dict(),
            'scaler': scaler.state_dict(),
            "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'torch_random': torch.random.get_rng_state(),
            'numpy_random': np.random.get_state(),
            'python_random': random.getstate()
        }

        torch.save(checkpoint, current_checkpoint_path)

        pd.DataFrame(data=history_record).to_csv(record_path, index=True)

        if early_stop.early_stop:
            break

        if is_increase:
            epoch_pbar.write(
                wrapper_text(
                    f"trigger counter: {early_stop.counter}/{early_stop.patience}",
                    "debug")
            )

    print("\nTraining Finished!")
