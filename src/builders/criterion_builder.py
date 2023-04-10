import torch.nn as nn


def build(config, mode):
    criterion = dict()
    criterion["classification"] = nn.CrossEntropyLoss()
    return criterion
