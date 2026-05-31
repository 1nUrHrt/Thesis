import os

import torch


class EarlyStopping:
    def __init__(self, save_path, patience=5, min_delta=1e-4, mode='min'):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        self.save_path = save_path
        if patience < 0:
            raise ValueError(f"patience不能为负数，当前值: {patience}")
        self.patience = patience
        if mode not in ['min', 'max']:
            raise ValueError(f"mode必须是'min'或max'，当前值: {mode}")
        self.mode = mode

        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_metric_val = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, metric_value, model_dicts):
        if self.mode == 'min':
            score = -metric_value
        else:
            score = metric_value

        is_increase = False

        if self.best_score is None:
            self.best_score = score
            self.best_metric_val = metric_value
            self.save_checkpoint(model_dicts)
            self.early_stop = False

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            is_increase = True
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_metric_val = metric_value
            self.save_checkpoint(model_dicts)
            self.counter = 0
            self.early_stop = False

        return is_increase

    def state_dict(self):
        state = {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_metric_val': self.best_metric_val,
            'early_stop': self.early_stop,
        }
        return state

    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_metric_val = state_dict['best_metric_val']
        self.early_stop = state_dict['early_stop']

    def save_checkpoint(self, model_dicts):
        model_dicts['best_score'] = self.best_score
        model_dicts['best_metric_val'] = self.best_metric_val
        torch.save(model_dicts, self.save_path)
