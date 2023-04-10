import logging
from src.core.meters import AverageEpochMeter


def build() -> dict:

    loss_meters = {
        "main_loss": AverageEpochMeter("Main Loss")}

    return loss_meters
