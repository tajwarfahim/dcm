import torch
from utils.pytorch_pairwise_dataset import (
    load_dataset,
    SampledDataset,
    SelectiveClassDataset,
)


def create_train_loader(args):
    id_train_dataset = load_dataset(
        dataset_name=args["id_dataset_name"],
        dataset_path=args["id_dataset_path"],
        train=True,
        id_dataset_name=args["id_dataset_name"],
        id_dataset_path=args["id_dataset_path"],
        augment=True,
    )

    id_train_dataset = SampledDataset(
        dataset=id_train_dataset,
        start_index=args["id_train_start_index"],
        end_index=args["id_train_end_index"],
    )

    id_train_dataset = SelectiveClassDataset(
        dataset=id_train_dataset,
        start_label=args["id_dataset_start_label"],
        end_label=args["id_dataset_end_label"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=id_train_dataset,
        batch_size=args["train_batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
    )

    _, mean, std = load_dataset(
        dataset_name=args["id_dataset_name"],
        dataset_path=args["id_dataset_path"],
        train=True,
        id_dataset_name=args["id_dataset_name"],
        id_dataset_path=args["id_dataset_path"],
        augment=False,
        return_mean=True,
    )

    return train_dataloader, mean, std


def create_val_dataloader(args):
    id_val_dataset = load_dataset(
        dataset_name=args["id_dataset_name"],
        dataset_path=args["id_dataset_path"],
        train=args["val_is_not_train"],
        id_dataset_name=args["id_dataset_name"],
        id_dataset_path=args["id_dataset_path"],
        augment=False,
    )

    id_val_dataset = SampledDataset(
        dataset=id_val_dataset,
        start_index=args["id_val_start_index"],
        end_index=args["id_val_end_index"],
    )

    id_val_dataset = SelectiveClassDataset(
        dataset=id_val_dataset,
        start_label=args["id_dataset_start_label"],
        end_label=args["id_dataset_end_label"],
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=id_val_dataset,
        batch_size=args["val_batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    return validation_dataloader


def create_test_dataloader(args):
    id_test_dataset = load_dataset(
        dataset_name=args["id_dataset_name"],
        dataset_path=args["id_dataset_path"],
        train=False,
        id_dataset_name=args["id_dataset_name"],
        id_dataset_path=args["id_dataset_path"],
        augment=False,
    )

    id_test_dataset = SampledDataset(
        dataset=id_test_dataset,
        start_index=args["id_test_start_index"],
        end_index=args["id_test_end_index"],
    )

    id_test_dataset = SelectiveClassDataset(
        dataset=id_test_dataset,
        start_label=args["id_dataset_start_label"],
        end_label=args["id_dataset_end_label"],
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=id_test_dataset,
        batch_size=args["val_batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    return test_dataloader


def create_ood_dataloader(args):
    ood_dataset = load_dataset(
        dataset_name=args["ood_dataset_name"],
        dataset_path=args["ood_dataset_path"],
        train=False,
        id_dataset_name=args["id_dataset_name"],
        id_dataset_path=args["id_dataset_path"],
        augment=False,
    )

    ood_dataset = SampledDataset(
        dataset=ood_dataset,
        start_index=args["ood_test_start_index"],
        end_index=args["ood_test_end_index"],
    )

    ood_dataset = SelectiveClassDataset(
        dataset=ood_dataset,
        start_label=args["ood_dataset_start_label"],
        end_label=args["ood_dataset_end_label"],
    )

    ood_dataloader = torch.utils.data.DataLoader(
        dataset=ood_dataset,
        batch_size=args["val_batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    return ood_dataloader


def create_unlabeled_dataloader(args):
    id_test_dataset = load_dataset(
        dataset_name=args["id_dataset_name"],
        dataset_path=args["id_dataset_path"],
        train=False,
        id_dataset_name=args["id_dataset_name"],
        id_dataset_path=args["id_dataset_path"],
        augment=True,
    )

    id_test_dataset = SampledDataset(
        dataset=id_test_dataset,
        start_index=args["id_unlabeled_start_index"],
        end_index=args["id_unlabeled_end_index"],
    )

    id_test_dataset = SelectiveClassDataset(
        dataset=id_test_dataset,
        start_label=args["id_dataset_start_label"],
        end_label=args["id_dataset_end_label"],
    )

    print("Len ID unlabeled set: ", len(id_test_dataset))

    ood_test_dataset = load_dataset(
        dataset_name=args["ood_dataset_name"],
        dataset_path=args["ood_dataset_path"],
        train=False,
        id_dataset_name=args["id_dataset_name"],
        id_dataset_path=args["id_dataset_path"],
        augment=True,
    )

    ood_test_dataset = SampledDataset(
        dataset=ood_test_dataset,
        start_index=args["ood_unlabeled_start_index"],
        end_index=args["ood_unlabeled_end_index"],
    )

    ood_test_dataset = SelectiveClassDataset(
        dataset=ood_test_dataset,
        start_label=args["ood_dataset_start_label"],
        end_label=args["ood_dataset_end_label"],
    )

    print("Len OOD unlabeled set: ", len(ood_test_dataset))

    unlabeled_dataset = torch.utils.data.ConcatDataset(
        datasets=[id_test_dataset, ood_test_dataset]
    )
    unlabeled_dataset = torch.utils.data.ConcatDataset(
        datasets=[unlabeled_dataset for i in range(args["repeat_unlabeled_set"])]
    )

    unlabeled_dataloader = torch.utils.data.DataLoader(
        dataset=unlabeled_dataset,
        batch_size=args["train_batch_size"] * 2,
        shuffle=True,
        num_workers=args["num_workers"],
    )

    return unlabeled_dataloader


def create_dataloaders(args):
    dataloaders = {}
    # get id train
    dataloaders["train"], dataloaders["mean"], dataloaders["std"] = create_train_loader(args=args)

    # get id val
    dataloaders["val"] = create_val_dataloader(args=args)

    # get id test
    dataloaders["test"] = create_test_dataloader(args=args)

    # get ood dataloader
    dataloaders["ood"] = create_ood_dataloader(args=args)

    # get unlabeled dataloader
    dataloaders["unlabeled"] = create_unlabeled_dataloader(args=args)

    print("\nID train num examples: ", len(dataloaders["train"].dataset))
    print("ID val num examples: ", len(dataloaders["val"].dataset))
    print("ID test num examples: ", len(dataloaders["test"].dataset))
    print("OOD test num examples: ", len(dataloaders["ood"].dataset))
    print("Unlabeled set num examples: ", len(dataloaders["unlabeled"].dataset), "\n")

    return dataloaders
