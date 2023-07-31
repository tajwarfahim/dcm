import os
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler


class CUB_TCM(Dataset):
    def __init__(self, data_dir, split, transform):
        assert split in ["train", "val1", "val2", "test_id", "test_ood"]
        full_dataset = CUBDataset(
            root_dir=data_dir,
            target_name="waterbird_complete95",
            confounder_names=["forest2water2"],
            model_type="resnet50",
            augment_data=False,
        )
        with open("./data_splits/cub_train.pkl", "rb") as f:
            train_indices_in = pickle.load(f)
        with open("./data_splits/cub_val1_in.pkl", "rb") as f:
            val_indices_in1 = pickle.load(f)
        with open("./data_splits/cub_val2_in.pkl", "rb") as f:
            val_indices_in2 = pickle.load(f)
        with open("./data_splits/cub_test_in.pkl", "rb") as f:
            test_indices_in = pickle.load(f)

        if split == "train":
            self.dataset = DRODataset(
                torch.utils.data.Subset(full_dataset, list(train_indices_in)),
                process_item_fn=None,
                n_groups=full_dataset.n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str,
            )
        elif split == "val1":
            self.dataset = DRODataset(
                torch.utils.data.Subset(full_dataset, list(val_indices_in1)),
                process_item_fn=None,
                n_groups=full_dataset.n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str,
            )
        elif split == "val2":
            self.dataset = DRODataset(
                torch.utils.data.Subset(full_dataset, list(val_indices_in2)),
                process_item_fn=None,
                n_groups=full_dataset.n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str,
            )
        elif split == "test_id":
            self.dataset = DRODataset(
                torch.utils.data.Subset(full_dataset, list(test_indices_in)),
                process_item_fn=None,
                n_groups=full_dataset.n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str,
            )
        elif split == "test_ood":
            dro_subsets = prepare_confounder_data(
                root_dir=data_dir,
                target_name="waterbird_complete95",
                confounder_names=["forest2water2"],
                model="resnet50",
                augment_data=False,
                splits=["train", "val", "test"],
                return_full_dataset=False,
            )
            self.dataset = dro_subsets[2]
        self.dataset.transform = transform

    def __getitem__(self, index):
        img, target, _, _ = self.dataset[index]

        return img, target, index

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return "waterbirds"


########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(
    root_dir,
    target_name,
    confounder_names,
    model,
    augment_data,
    splits,
    return_full_dataset=False,
):
    full_dataset = CUBDataset(
        root_dir=root_dir,
        target_name=target_name,
        confounder_names=confounder_names,
        model_type=model,
        augment_data=augment_data,
    )
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
    subsets = full_dataset.get_splits(splits, train_frac=1.0)
    dro_subsets = [
        DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
        for split in splits
    ]
    return dro_subsets


class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        group_array = []
        y_array = []

        for x, y, g, _ in self:
            group_array.append(g)
            y_array.append(y)
        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (
            (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()
        )
        self._y_counts = (
            (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()
        )

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, y, g in self:
            return x.size()

    def get_loader(self, train, reweight_groups, **kwargs):
        if not train:  # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups:  # Training but not reweighting
            shuffle = True
            sampler = None
        else:  # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self) / self._group_counts
            weights = group_weights[self._group_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(self, shuffle=shuffle, sampler=sampler, **kwargs)
        return


class ConfounderDataset(Dataset):
    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names,
        model_type=None,
        augment_data=None,
    ):
        raise NotImplementedError

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        if model_attributes[self.model_type]["feature_type"] == "precomputed":
            x = self.features_mat[idx, :]
        else:
            img_filename = os.path.join(self.data_dir, self.filename_array[idx])
            img = Image.open(img_filename).convert("RGB")
            # Figure out split and transform accordingly
            if self.split_array[idx] == self.split_dict["train"] and self.train_transform:
                img = self.train_transform(img)
            elif (
                self.split_array[idx] in [self.split_dict["val"], self.split_dict["test"]]
                and self.eval_transform
            ):
                img = self.eval_transform(img)
            # Flatten if needed
            if model_attributes[self.model_type]["flatten"]:
                assert img.dim() == 3
                img = img.view(-1)
            x = img

        return x, y, g, idx

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ("train", "val", "test"), f"{split} is not a valid split"
            mask = self.split_array == self.split_dict[split]

            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == "train":
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f"{self.target_name} = {int(y)}"
        bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f", {attr_name} = {bin_str[attr_idx]}"
        return group_name


class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """

    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names,
        augment_data=False,
        model_type=None,
        metadata_csv_name="metadata.csv",
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        self.data_dir = os.path.join(
            self.root_dir, "data", "_".join([self.target_name] + self.confounder_names)
        )
        print(os.path.abspath(self.data_dir))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        # Read in metadata
        print(f"Reading '{os.path.join(self.data_dir, metadata_csv_name)}'")
        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, metadata_csv_name))

        # Get the y values
        self.y_array = self.metadata_df["y"].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df["place"].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        assert self.n_groups == 4, "check the code if you are running otherwise"
        self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype(
            "int"
        )

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        # Set transform
        if model_attributes[self.model_type]["feature_type"] == "precomputed":
            self.features_mat = torch.from_numpy(
                np.load(
                    os.path.join(
                        root_dir,
                        "features",
                        model_attributes[self.model_type]["feature_filename"],
                    )
                )
            ).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_cub(
                self.model_type, train=True, augment_data=augment_data
            )
            self.eval_transform = get_transform_cub(
                self.model_type, train=False, augment_data=augment_data
            )


def get_transform_cub(model_type, train, augment_data):
    scale = 256.0 / 224.0
    target_resolution = model_attributes[model_type]["target_resolution"]
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(target_resolution[0] * scale),
                        int(target_resolution[1] * scale),
                    )
                ),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transform


model_attributes = {
    "bert": {"feature_type": "text"},
    "inception_v3": {
        "feature_type": "image",
        "target_resolution": (299, 299),
        "flatten": False,
    },
    "wideresnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet34": {"feature_type": "image", "target_resolution": None, "flatten": False},
    "raw_logistic_regression": {
        "feature_type": "image",
        "target_resolution": None,
        "flatten": True,
    },
    "bert-base-uncased": {"feature_type": "text"},
}
