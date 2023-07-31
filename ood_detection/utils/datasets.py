import torch
from collections import defaultdict


class UnionDataset(torch.utils.data.Dataset):

    """
    Takes multiple datasets and constructs their union.
    Note: The labels of all datasets are remapped to be in consecutive order.
    (i.e. the same label in two datasets will be to two different labels in the union)

    Parameters:
        datasets (list of SampledDataset) -  A list of SampledDataset objects.
    ~~~
    """

    def _get_class_to_idx_mapping(self, dataset):
        class_to_idx_mapping = defaultdict(list)
        for index in range(len(dataset)):
            _, label = dataset[index]
            if torch.is_tensor(label):
                label = label.item()

            class_to_idx_mapping[label].append(index)

        return class_to_idx_mapping

    def __init__(self, datasets):
        datasets = [ds for ds in datasets]
        self.datasets = datasets
        labels_to_indices = defaultdict(list)
        indices_to_labels = defaultdict(int)
        labels_count = 0
        indices_count = 0
        for dataset in datasets:
            dataset_class_to_idx_mapping = self._get_class_to_idx_mapping(dataset)
            dataset_classes = sorted(list(dataset_class_to_idx_mapping.keys()))
            labels_nooffset = {label: i for i, label in enumerate(dataset_classes)}
            for label, indices in dataset_class_to_idx_mapping.items():
                label = labels_nooffset[label]
                for idx in indices:
                    indices_to_labels[indices_count + idx] = labels_count + label
                    labels_to_indices[labels_count + label].append(indices_count + idx)
            indices_count += len(dataset)
            labels_count += len(dataset_classes)

        self.indices_count = indices_count
        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

    def __getitem__(self, item):
        ds_count = 0
        for dataset in self.datasets:
            if ds_count + len(dataset) > item:
                img, label = dataset[item - ds_count]
                label = self.indices_to_labels[item]
                return img, label
            ds_count += len(dataset)

    def __len__(self):
        return len(self.indices_to_labels)
