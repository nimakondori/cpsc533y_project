from torch.optim import Adam


def build(model, config):
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer
