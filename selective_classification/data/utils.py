import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unpack_example(example):
    return example[0], example[1], example[2]


def get_misclassified_examples(net, loader):
    assert loader.batch_size == 1
    misclassified_idxs = []
    net.eval()
    with torch.no_grad():
        for example_idx, example in enumerate(loader):
            data, target, _ = unpack_example(example)
            data, target = data.to(device), target.to(device)
            output = net(data)
            pred = torch.argmax(F.softmax(output, dim=-1), dim=-1)
            if not pred == target:
                misclassified_idxs.append(example_idx)

    return misclassified_idxs


class MisclassifiedDataset(Dataset):
    def __init__(self, dataset, model, transform=None, collate_fn=None):
        """
        Parameters
        ----------
        dataset
        start_index: start index of examples in each class
        end_index: end index of examples in each class
        """
        self.model = model
        self.dataset = dataset
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        self.indices = get_misclassified_examples(model, dataloader)
        self.dataset.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        remapped_index = self.indices[index]
        data, target, _ = unpack_example(self.dataset[remapped_index])
        return data, target, index


class CorrectDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, model, transform=None, collate_fn=None):
        """
        Parameters
        ----------
        dataset
        start_index: start index of examples in each class
        end_index: end index of examples in each class
        """
        self.model = model
        self.dataset = dataset
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        misclassified_indices = get_misclassified_examples(model, dataloader)
        all_indices = np.arange(len(dataset))
        self.indices = all_indices[~np.isin(all_indices, misclassified_indices)]
        self.dataset.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        remapped_index = self.indices[index]
        data, target, _ = unpack_example(self.dataset[remapped_index])
        return data, target, index


class CustomSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, target, _ = unpack_example(self.dataset[self.indices[index]])
        if self.transform:
            img = self.transform(img)
        return img, target, index


class SampledDataset(Dataset):
    def _get_class_to_idx_mapping(self, dataset):
        class_to_idx_mapping = collections.defaultdict(list)
        for index in range(len(dataset)):
            _, label, _ = dataset[index]
            if torch.is_tensor(label):
                label = label.item()

            class_to_idx_mapping[label].append(index)

        return class_to_idx_mapping

    def __init__(
        self,
        dataset,
        start_index,
        end_index,
    ):
        """
        Parameters
        ----------
        dataset
        start_index: start index of examples in each class
        end_index: end index of examples in each class
        """
        self.dataset = dataset
        self.class_to_idx_mapping = self._get_class_to_idx_mapping(dataset=self.dataset)

        indices = []
        for class_label in self.class_to_idx_mapping.keys():
            class_indices = self.class_to_idx_mapping[class_label]
            indices += class_indices[start_index:end_index]

        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        remapped_index = self.indices[index]
        return self.dataset[remapped_index][0], self.dataset[remapped_index][1], index

    def get_raw_data(self, index):
        remapped_index = self.indices[index]
        return self.dataset.data[remapped_index]


class BinaryDataset(Dataset):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )
        self.misclassified_indices = get_misclassified_examples(model, dataloader)
        self.targets = self._get_targets()

    def __getitem__(self, index):
        image, _, _ = unpack_example(self.dataset[index])
        if index in self.misclassified_indices:
            label = 1
        else:
            label = 0
        return image, label, index

    def __len__(self):
        return len(self.dataset)

    def _get_targets(self):
        targets = []
        for idx in range(len(self.dataset)):
            if idx in self.misclassified_indices:
                targets.append(1)
            else:
                targets.append(0)
        return targets
