import os

import torch


class EarlyStopping:
    def __init__(self, save_path, patience=5, min_delta=1e-4, mode="min"):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        self.save_path = save_path
        if patience < 0:
            raise ValueError(f"patience不能为负数,当前值: {patience}")
        self.patience = patience
        if mode not in ["min", "max"]:
            raise ValueError(f"mode必须是'min'或max'，当前值: {mode}")
        self.mode = mode

        self.min_delta = min_delta
        self.counter = 0
        self.best_metric_val = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, metric_value, epoch, model_dicts):
        if self.best_metric_val is None:
            is_improved = True
        else:
            if self.mode == "min":
                is_improved = metric_value < self.best_metric_val - self.min_delta
            else:
                is_improved = metric_value > self.best_metric_val + self.min_delta

        if is_improved:
            self.counter = 0
            self.best_epoch = epoch
            self.best_metric_val = metric_value
            self.early_stop = False
            self.save_checkpoint(model_dicts)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return is_improved


    def state_dict(self):
        state = {
            "counter": self.counter,
            "best_epoch": self.best_epoch,
            "best_metric_val": self.best_metric_val,
            "early_stop": self.early_stop,
        }
        return state

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]
        self.best_epoch = state_dict["best_epoch"]
        self.best_metric_val = state_dict["best_metric_val"]
        self.early_stop = state_dict["early_stop"]

    def save_checkpoint(self, model_dicts):
        model_dicts["counter"] = self.counter
        model_dicts["best_epoch"] = self.best_epoch
        model_dicts["best_metric_val"] = self.best_metric_val
        model_dicts["early_stop"] = self.early_stop
        torch.save(model_dicts, self.save_path)
