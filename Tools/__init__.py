from .InteractionDataset import InteractionDataset
from .DrugDataset import DrugDataset
from .EarlyStopping import EarlyStopping

COLORS = {
    'info': '\033[92m',  # 绿色
    'debug': '\033[94m',  # 蓝色
    'warning': '\033[93m',  # 黄色
    'error': '\033[91m',  # 红色
    'reset': '\033[0m'
}


def wrapper_text(text, mode):
    return f"{COLORS[mode]}{text}{COLORS['reset']}"


__all__ = ["InteractionDataset", "DrugDataset", "EarlyStopping"]