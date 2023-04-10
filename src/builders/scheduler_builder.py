from torch.optim import lr_scheduler


def build(optimizer, config):

    return lr_scheduler.ReduceLROnPlateau(
        optimizer,
        # TODO: Check the effectiveness of this scheduler and add configs for it
        factor=0.5,
        patience=config.patience,
        threshold=config.threshold,
        min_lr=config.min_lr,
    )
