from src.core.data import AorticStenosisDataset
from torch.utils.data import DataLoader
# from torch import distributed as dist
import os
import torch
import platform

DATASETS = {
    "as": AorticStenosisDataset,
}


def get_dataloaders(config, dataset_train, dataset_val, train=True):
    dataloaders = dict()

    if platform.system() == "Windows":
        # Set the number of workers to 0 on Windows to avoid issues with DataLoader
        num_workers = 0
    else:
        num_workers = min(8, os.cpu_count())
        
    if train:
        dataloaders.update(
            {
                "train": DataLoader(
                    dataset_train,
                    batch_size=config["batch_size"],
                    sampler=dataset_train.class_samplers(),
                    num_workers=num_workers,
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
                # batch_size=config["batch_size"],
                shuffle=False,
                num_workers=1,
                # num_workers=min(8, os.cpu_count()),
                pin_memory=True,
                drop_last=False,
                # collate_fn=collate_fn
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
            use_metadata=config.use_metadata,
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
        use_metadata=config.use_metadata,
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

# def collate_fn(batch):
#     # Get the maximum sequence length in the batch
#     max_len = max([len(x) for x in batch])

#     # Pad the sequences with zeros to the same length
#     padded = [torch.nn.functional.pad(x['vid'], (0, max_len - len(x['vid']))) for x in batch]

#     # Stack the padded sequences into a batch tensor
#     batch_tensor = torch.stack(padded, dim=0)
    # return batch_tensor

def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if  key in ["vid", "mask"]:
            ndim = batch[0][key].ndim
            max_len = max([len(sample[key]) for sample in batch])
            # padded = torch.stack([torch.nn.functional.pad(sample[key], (max_len - len(sample[key]), 0)) for sample in batch], dim=0)
            # Repeat the first dimension of all the items in the batch to match the largest item in the batch
            padded = torch.stack([sample[key].expand((max_len, ) + (-1, )*(ndim - 1)) for sample in batch])
            collated[key] = padded
        elif torch.is_tensor(batch[0][key]):
            collated[key] = torch.stack([sample[key] for sample in batch])
        else:
            collated[key] = torch.stack([sample[key] for sample in batch])
    return collated