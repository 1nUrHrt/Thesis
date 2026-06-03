from AttnEncoder import AttnEncoder
from AttnResEncoder import AttnResEncoder
from GeneralModel import Decoder
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
    if model_name == "Decoder":
        return Decoder(**kwargs)
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