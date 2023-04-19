import collections
import itertools
import math
import numpy as np
from os.path import join
import pandas as pd
import random
from scipy.io import loadmat
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

# If we call see everyhting, should we remove this?
random.seed(0)
np.random.seed(0)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)

label_schemes = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}


def bicuspid_filter(df: pd.DataFrame):
    return df[~df["Bicuspid"]]


filtering_functions: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "bicuspid": bicuspid_filter
}


class AorticStenosisDataset(Dataset):
    def __init__(
        self,
        dataset_path: str = "~/as",
        split: str = "train",
        mode: str = "as",
        max_frames: int = 16,
        transform=None,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
        use_metadata=False,
    ):

        super().__init__()

        assert mode == "as", "Only AS mode is supported"

        # navigation for linux environment
        # dataset_root = dataset_root.replace('~', os.environ['HOME'])

        # read in the data directory CSV as a pandas dataframe
        dataset = pd.read_csv(join(dataset_path, "annotations-all_short.csv"))
        # append dataset root to each path in the dataframe
        dataset["path"] = dataset["path"].map(lambda x: join(dataset_path, x))

        dataset = dataset[dataset["as_label"].map(lambda x: x in label_schemes.keys())]

        # Take train/test/val
        dataset = dataset[dataset.split == split]
        # Apply an arbitrary filter
        # filtering_function = filtering_functions["bicuspid"]
        # dataset = filtering_function(dataset)

        self.patient_studies = list()
        for patient_id in list(set(dataset["patient_id"])):
            for study_date in list(
                set(dataset[dataset["patient_id"] == patient_id]["date"])
            ):
                self.patient_studies.append((patient_id, study_date))

        self.dataset = dataset
        self.max_frames = max_frames
        self.train = split == "train"
        self.trans = transform
        self.aug_trans = aug_transform
        self.max_clips = max_clips
        self.mean_std = mean_std
        self.use_metadata = use_metadata
        if self.use_metadata:
            self.metadata = pd.read_csv(join(dataset_path, "annotations-all_meta_cleaned.csv"))

    def class_samplers(self):
        labels_AS = list()
        for pid in self.patient_studies:
            patient_id, study_date = pid
            data_info = self.dataset[self.dataset["patient_id"] == patient_id]
            data_info = data_info[data_info["date"] == study_date]

            labels_AS.append(label_schemes[data_info["as_label"].iloc[0]])

        class_sample_count_AS = np.array(
            [len(np.where(labels_AS == t)[0]) for t in np.unique(labels_AS)]
        )
        weight_AS = 1.0 / class_sample_count_AS

        # This in case we only use 3 categories. We adde a 0 at position 0
        if len(weight_AS) != 4:
            weight_AS = np.insert(weight_AS, 0, 0)
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()

        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))

        return sampler_AS

    def __len__(self) -> int:
        return len(self.patient_studies)

    def __getitem__(self, item):
        patient_id, study_date = self.patient_studies[item]
        data_info = self.dataset[self.dataset["patient_id"] == patient_id]
        data_info = data_info[data_info["date"] == study_date]
        # label = torch.tensor(self.labelling_scheme[data_info[self.label_key]])

        available_views = list(data_info["view"])
        num_plax = available_views.count("plax")
        num_psax = available_views.count("psax")

        frame_nums = list()

        if self.mean_std:
            return self.trans(
                np.moveaxis(loadmat(data_info["path"].iloc[0])["cine"], 0, -1)
            )

        all_plax_cine = list()
        if num_plax > 0:
            plax_indices = (
                [np.random.randint(num_plax)] if self.train else list(range(num_plax))
            )

            for plax_idx in plax_indices:
                plax_data_info = data_info[data_info["view"] == "plax"].iloc[plax_idx]

                # Transform and augment PLAX vid
                plax_cine = self.trans(
                    np.moveaxis(loadmat(plax_data_info["path"])["cine"], 0, -1)
                )
                if self.aug_trans is not None:
                    plax_cine = plax_cine.unsqueeze(0)
                    plax_cine = self.aug_trans(plax_cine)
                    plax_cine = plax_cine.squeeze(0)

                all_plax_cine.append(plax_cine)
                frame_nums.append(plax_cine.shape[0])

        all_psax_cine = list()
        if num_psax > 0:
            psax_indices = (
                [np.random.randint(num_psax)] if self.train else list(range(num_psax))
            )

            for psax_idx in psax_indices:
                psax_data_info = data_info[data_info["view"] == "psax"].iloc[psax_idx]

                # Transform and augment psax vid
                psax_cine = self.trans(
                    np.moveaxis(loadmat(psax_data_info["path"])["cine"], 0, -1)
                )
                if self.aug_trans is not None:
                    psax_cine = psax_cine.unsqueeze(0)
                    psax_cine = self.aug_trans(psax_cine)
                    psax_cine = psax_cine.squeeze(0)

                all_psax_cine.append(psax_cine)
                frame_nums.append(psax_cine.shape[0])

        no_plax = False
        no_psax = False
        if num_plax == 0:
            all_plax_cine.append(torch.zeros_like(all_psax_cine[0]))
            num_plax = 1
            no_plax = True
        elif num_psax == 0:
            all_psax_cine.append(torch.zeros_like(all_plax_cine[0]))
            num_psax = 1
            no_psax = True

        if not self.train:
            # Extract all possible clips in the
            num_clips = min(
                math.ceil(max(frame_nums) / self.max_frames),
                self.max_clips,
            )

            plax_psax_comb = list(
                itertools.product(list(range(num_plax)), list(range(num_psax)))
            )

            if len(plax_psax_comb) > 6:
                plax_psax_comb = plax_psax_comb[:6]

            mask = torch.ones(
                (
                    num_clips * len(plax_psax_comb),
                    2,
                    self.max_frames,
                ),
                dtype=torch.bool,
            )

            plax_temp = list()
            psax_temp = list()

            for combination_idx in range(len(plax_psax_comb)):
                for clip_idx in range(num_clips):
                    (
                        plax_cine_temp,
                        mask[(num_clips * combination_idx) + clip_idx],
                    ) = self.pad_vid(
                        all_plax_cine[plax_psax_comb[combination_idx][0]],
                        mask[(num_clips * combination_idx) + clip_idx],
                        0,
                        clip_idx,
                    )
                    plax_temp.append(plax_cine_temp.unsqueeze(0))

                    (
                        psax_cine_temp,
                        mask[(num_clips * combination_idx) + clip_idx],
                    ) = self.pad_vid(
                        all_psax_cine[plax_psax_comb[combination_idx][1]],
                        mask[(num_clips * combination_idx) + clip_idx],
                        1,
                        clip_idx,
                    )
                    psax_temp.append(psax_cine_temp.unsqueeze(0))

            plax_cine = torch.cat(plax_temp, dim=0)
            psax_cine = torch.cat(psax_temp, dim=0)
            cine = torch.cat(
                (
                    plax_cine.unsqueeze(1).unsqueeze(-1),
                    psax_cine.unsqueeze(1).unsqueeze(-1),
                ),
                dim=1,
            )

            if no_plax:
                mask[:, 0, :] = False
            elif no_psax:
                mask[:, 1, :] = False
        else:
            mask = torch.ones((2, self.max_frames), dtype=torch.bool)

            plax_cine, mask = self.pad_vid(all_plax_cine[0], mask, 0)
            psax_cine, mask = self.pad_vid(all_psax_cine[0], mask, 1)

            cine = torch.cat(
                (
                    plax_cine.unsqueeze(0).unsqueeze(-1),
                    psax_cine.unsqueeze(0).unsqueeze(-1),
                ),
                dim=0,
            )

            if no_plax:
                mask[0, :] = False
            elif no_psax:
                mask[1, :] = False

        label = label_schemes[data_info["as_label"].iloc[0]]
        if self.use_metadata:
            metadata = self.metadata[self.metadata["patient_id"] == patient_id]
            # Grab the most recent metadata and drop the patient id, and date
            metadata = torch.tensor(metadata.sort_values(by='date', ascending=False).drop(
                ["patient_id", "date"], axis=1).iloc[0].values)
            # iloc [0] is for studies that are duplicated
            # metadata = torch.tensor((metadata[metadata["date"] == study_date]).drop(["patient_id", "date"], axis=1).iloc[0].values)
            # Create point cloud 
            metadata_pc, pc_edge_index, pc_features = self.create_point_cloud(metadata)

        # Create a fully connected graph with the video frames as nodes

        gnn_edge_index = torch.combinations(torch.arange(cine.shape[1] if self.train else cine.shape[2]), with_replacement=True).T
        gnn_edge_index = gnn_edge_index[:, gnn_edge_index[0] != gnn_edge_index[1]]
                    
        # When not self.train, the evaluation grabs all the possible clips form the video
        return {
            "vid": cine.repeat(1, 1, 1, 1, 3) if self.train else cine.repeat(1, 1, 1, 1, 1, 3), # convert to 3 channels, 
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "metadata": metadata_pc,
            "pc_edge_index": pc_edge_index,
            "pc_features": pc_features,
            "gnn_edge_index": gnn_edge_index,
            # "class_label": torch.zeros(1),
        }

    def pad_vid(self, vid, mask, mask_idx, clip_idx=0):
        if vid.shape[0] < self.max_frames:
            mask[mask_idx, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)
                vid = vid[starting_idx : starting_idx + self.max_frames]
            else:
                if (clip_idx + 1) * self.max_frames <= vid.shape[0]:
                    vid = vid[
                        self.max_frames * clip_idx : self.max_frames * (clip_idx + 1)
                    ]
                else:
                    vid = vid[-self.max_frames :]

        return vid, mask
    

    def create_point_cloud(self, metadata):
        # make and mxm point cloud here for the projection maps
        # where m is the number of metadata features
        metadata = metadata.repeat(len(metadata), 1)
        for i in range(len(metadata)):
            # set the diagonal to 0
            metadata[i, i] = 0

        # Create a fully connected graph with edge_index list for the point cloud
        edge_index = torch.combinations(torch.arange(len(metadata)), with_replacement=True).T
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        # Create pc features which is just a tensor of ones 
        pc_features = torch.ones(metadata.shape)

        return metadata, edge_index, pc_features