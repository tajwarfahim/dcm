import torchvision.transforms as transforms
import numpy as np
from robustbench.data import load_cifar10c, load_cifar100c
from torch.utils.data import Dataset

cifar_10c_splits = {"train": [0, 5000], "val": [5000, 6000], "test": [6000, 10000]}


def get_split_indices(targets, start_idx, end_idx):
    num_classes = int(max(targets)) + 1
    labels = {}
    for i in range(num_classes):
        labels[i] = [ind for ind, n in enumerate(targets) if n == i]  # np.where()

    indices = []
    for i in range(len(labels.keys())):
        np.random.shuffle(labels[i])
        indices.append(labels[i][start_idx:end_idx])
    indices = np.concatenate(indices)

    return indices


class CIFAR10C(Dataset):
    def __init__(
        self, data_dir, split, corruption_type, severity, transform, train_set_size=None
    ):
        assert split in ["train", "val", "test"]
        n_examples = cifar_10c_splits["test"][1]
        x_corr, y_corr = load_cifar10c(
            n_examples=n_examples,
            severity=severity,
            data_dir=data_dir,
            shuffle=False,
            corruptions=[corruption_type],
        )
        self.data = x_corr
        self.targets = y_corr
        self.num_classes = int(max(self.targets)) + 1
        if split == "train":
            start_idx = 0
            end_idx = cifar_10c_splits["train"][1] // self.num_classes
        elif split == "val":
            start_idx = cifar_10c_splits["train"][1] // self.num_classes
            end_idx = cifar_10c_splits["val"][1] // self.num_classes
        else:
            start_idx = cifar_10c_splits["val"][1] // self.num_classes
            end_idx = cifar_10c_splits["test"][1] // self.num_classes
        self.indices = get_split_indices(targets=y_corr, start_idx=start_idx, end_idx=end_idx)
        self.transform = transform

    def __getitem__(self, index):
        index = self.indices[index]
        img, target = self.data[index], self.targets[index].item()

        if self.transform is not None:
            to_PIL = transforms.ToPILImage()
            img = to_PIL(img)
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.indices)

    def __str__(self):
        return "cifar10c"
