import os
import pickle

from torchvision import transforms

from data.camelyon17 import Camelyon17Dataset
from data.cifar import CIFAR10
from data.cifar_c_dataset import CIFAR10C
from data.cub_dataset import CUB_TCM, get_transform_cub
from data.fmow import FMoWDataset
from data.utils import CustomSubset, SampledDataset


def get_dataset(
    task_name, data_dir, train_transform, val_transform, test_transform, corruption_type=None
):
    if "cifar10" in task_name:
        assert corruption_type is not None, "Specify corruption type for CIFAR-10C."
        dataset_path = os.path.join(data_dir, "CIFAR10")
        train_data = CIFAR10(dataset_path, train=True, transform=train_transform)
        test_data_in = CIFAR10(dataset_path, train=False, transform=val_transform)
        val_data1 = SampledDataset(dataset=test_data_in, start_index=0, end_index=200)
        val_data2 = SampledDataset(dataset=test_data_in, start_index=500, end_index=600)
        test_data_in = SampledDataset(dataset=test_data_in, start_index=600, end_index=1000)
        test_data_ood = CIFAR10C(
            data_dir=data_dir,
            split="test",
            corruption_type=corruption_type,
            severity=5,
            transform=test_transform,
        )

    elif "waterbirds" in task_name:
        train_data = CUB_TCM(data_dir=data_dir, split="train", transform=train_transform)
        val_data1 = CUB_TCM(data_dir=data_dir, split="val1", transform=val_transform)
        val_data2 = CUB_TCM(data_dir=data_dir, split="val2", transform=val_transform)
        test_data_in = CUB_TCM(data_dir=data_dir, split="test_id", transform=test_transform)
        test_data_ood = CUB_TCM(data_dir=data_dir, split="test_ood", transform=test_transform)

    elif "camelyon17" in task_name:
        dataset = Camelyon17Dataset(version=None, download=True, root_dir=data_dir)
        train_data = dataset.get_subset("train", transform=train_transform)
        with open(f"./data_splits/camelyon17_val1_in.pkl", "rb") as f:
            val_indices_in1 = pickle.load(f)
        with open(f"./data_splits/camelyon17_val2_in.pkl", "rb") as f:
            val_indices_in2 = pickle.load(f)
        with open(f"./data_splits/camelyon17_test_in.pkl", "rb") as f:
            test_indices_in = pickle.load(f)
        val_data1 = CustomSubset(dataset, list(val_indices_in1), transform=test_transform)
        val_data2 = CustomSubset(dataset, list(val_indices_in2), transform=test_transform)
        test_data_in = CustomSubset(dataset, list(test_indices_in), transform=test_transform)
        test_data_ood = dataset.get_subset("test", transform=test_transform)

    elif "fmow" in task_name:
        dataset = FMoWDataset(version=None, download=True, root_dir=data_dir)
        train_data = dataset.get_subset("train", transform=train_transform)
        with open(f"./data_splits/fmow_val1_in.pkl", "rb") as f:
            val_indices1 = pickle.load(f)
        with open(f"./data_splits/fmow_val2_in.pkl", "rb") as f:
            val_indices2 = pickle.load(f)
        val_data1 = CustomSubset(dataset, list(val_indices1), transform=test_transform)
        val_data2 = CustomSubset(dataset, list(val_indices2), transform=test_transform)
        test_data_in = dataset.get_subset("id_test", transform=test_transform)
        test_data_ood = dataset.get_subset("test", transform=test_transform)

    return train_data, val_data1, val_data2, test_data_in, test_data_ood


def get_transforms(dataset_name):
    if dataset_name == "cifar10":
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()

    elif dataset_name == "cifar10c":
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.ToTensor()

    elif "waterbirds" in dataset_name:
        train_transform = get_transform_cub(
            model_type="resnet50", train=True, augment_data=True
        )
        test_transform = get_transform_cub(
            model_type="resnet50", train=False, augment_data=False
        )

    elif "camelyon17" in dataset_name:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    elif "fmow" in dataset_name:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = train_transform

    return train_transform, test_transform
