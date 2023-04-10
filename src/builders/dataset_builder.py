from src.core.data import AorticStenosisDataset
from torch.utils.data import DataLoader
from os.path import join
import os

DATASETS = {
    'aorticstenosis': AorticStenosisDataset,
}


def get_dataloaders(config, dataset_train, dataset_val, train=True):
    dataloaders = dict()

    if train:
        dataloaders.update(
            {
                "train": DataLoader(
                    dataset_train,
                    batch_size=config["batch_size"],
                    sampler=dataset_train.class_samplers(),
                    num_workers=min(8, os.cpu_count()),
                    pin_memory=True,
                    drop_last=True,
                )
            }
        )

    dataloaders.update(
        {
            "val": DataLoader(
                dataset_val,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
        }
    )

    return dataloaders


def build(config, train, transform, aug_transform, logger):
    dataset_name = config.name

    dataset_train = (
        DATASETS[dataset_name](
            dataset_path=config.dataset_path,
            mode=config.mode,
            max_frames=config.max_frames,
            transform=transform,
            aug_transform=aug_transform,
            split=config.split if config.mode == "pretrain" else "train",
            max_clips=config.max_clips,
        )
        if train
        else None
    )

    dataset_val = DATASETS[dataset_name](
        dataset_path=config.dataset_path,
        mode=config.mode,
        max_frames=config.max_frames,
        transform=transform,
        aug_transform=None,
        split="val" if train else "test",
        max_clips=config.max_clips,
    )

    dataloaders = get_dataloaders(config, dataset_train, dataset_val, train)

    if train:
        logger.info("Len of training dataset: {}".format(len(dataset_train)))
        logger.info("Len of validation dataset: {}".format(len(dataset_val)))

        print("Len of training dataset: {}".format(len(dataset_train)))
        print("Len of validation dataset: {}".format(len(dataset_val)))
    else:
        logger.info("Len of test dataset: {}".format(len(dataset_val)))
        print("Len of test dataset: {}".format(len(dataset_val)))

    return (
        dataloaders,
        None
        if dataset_name in ["as", "prostate_single_patch", "kinetics"]
        else dataset_val.patient_data_dirs,
    )
