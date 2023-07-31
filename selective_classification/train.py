import torch

from data.utils import MisclassifiedDataset, CorrectDataset, BinaryDataset
from datasets import get_dataset, get_transforms
from methods.binary_classifier import BinaryClassifier
from methods.dcm import DCM
from methods.dg import DG
from methods.ft import FT
from methods.sat import SAT
from methods.sat_loss import deep_gambler_loss, SelfAdaptiveTraining
from metrics import get_measures
from utils import cosine_annealing, collate_fn, get_checkpoint_path


def train(method, model, calibrator, config, corruption=None):
    # Datasets
    train_transform_id, test_transform_id = get_transforms(dataset_name=config.id_dataset)
    train_transform_ood, test_transform_ood = get_transforms(dataset_name=config.ood_dataset)
    train_data, val_data1, val_data2, test_data_id, test_data_ood = get_dataset(
        task_name=config.task_name,
        data_dir=config.data_dir,
        train_transform=train_transform_id,
        val_transform=test_transform_id,
        test_transform=test_transform_ood,
        corruption_type=corruption,
    )

    if method in ["msp", "ensemble", "max_logit"]:
        return model, calibrator, config.base_model_path

    # Uncertainty Set & Fine-tuning Set
    if method == "dcm":
        uncertainty_set = MisclassifiedDataset(
            dataset=val_data1, model=model, transform=train_transform_id, collate_fn=collate_fn
        )
        print("Number of Misclassified Validation Examples:", len(uncertainty_set))
        if config.repeat_uncertainty_set is not None:
            uncertainty_set_list = [
                uncertainty_set for _ in range(config.repeat_uncertainty_set)
            ]
            uncertainty_set = torch.utils.data.ConcatDataset(uncertainty_set_list)
        uncertainty_set_loader = torch.utils.data.DataLoader(
            uncertainty_set,
            batch_size=config.uncertainty_set_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        correct_set = CorrectDataset(
            dataset=val_data1, model=model, transform=train_transform_id, collate_fn=collate_fn
        )
        print("Number of Correct Validation Examples:", len(correct_set))
        finetune_set = torch.utils.data.ConcatDataset([train_data, correct_set])
    elif method == "bc":
        finetune_set = BinaryDataset(
            model, dataset=torch.utils.data.ConcatDataset([train_data, val_data1])
        )
    elif method == "ft":
        finetune_set = val_data1
        finetune_set.transform = train_transform_ood
    elif method in ["dg", "sat"]:
        finetune_set = val_data1

    # Criterion
    if method in ["dcm", "ft", "bc"]:
        criterion = torch.nn.functional.cross_entropy
    elif method == "dg":
        criterion = deep_gambler_loss
    elif method == "sat":
        criterion = SelfAdaptiveTraining(
            num_examples=len(finetune_set), num_classes=config.num_classes, mom=config.momentum
        )

    # Dataloaders
    finetuning_set_loader = torch.utils.data.DataLoader(
        finetune_set,
        batch_size=config.finetuning_set_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data2,
        batch_size=config.finetuning_set_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Optimizer and Scheduler
    if config.task in ["cifar10", "waterbirds", "fmow"]:
        optimizer = torch.optim.SGD(
            model.parameters(),
            config.lr,
            momentum=config.momentum,
            weight_decay=config.decay,
            nesterov=True,
        )
    elif config.task in ["camelyon17"]:
        optimizer = torch.optim.AdamW(model.parameters(), config.lr, **config.optimizer_config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            config.num_epochs * len(finetuning_set_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / config.lr,
        ),
    )

    # Trainer
    checkpoint_path = get_checkpoint_path(method=method, config=config)
    if method == "dcm":
        trainer = DCM(
            config=config,
            model=model,
            calibrator=None,
            finetune_set_loader=finetuning_set_loader,
            uncertainty_set_loader=uncertainty_set_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            setting=None,
            checkpoint_path=checkpoint_path,
            confidence_weight=config.confidence_weight,
        )
    elif method == "ft":
        trainer = FT(
            config=config,
            model=model,
            calibrator=None,
            finetuning_set_loader=finetuning_set_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            setting=None,
            checkpoint_path=checkpoint_path,
        )
    elif method == "bc":
        trainer = BinaryClassifier(
            config=config,
            model=model,
            calibrator=calibrator,
            finetuning_set_loader=finetuning_set_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            setting=None,
            checkpoint_path=checkpoint_path,
        )
    elif method == "dg":
        trainer = DG(
            config=config,
            model=model,
            calibrator=None,
            finetuning_set_loader=finetuning_set_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            setting=None,
            checkpoint_path=checkpoint_path,
        )
    elif method == "sat":
        trainer = SAT(
            config=config,
            model=model,
            calibrator=None,
            finetuning_set_loader=finetuning_set_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            setting=None,
            checkpoint_path=checkpoint_path,
        )

    print("Starting training...")
    trainer.run_training(num_epochs=config.num_epochs)
    checkpoint_path = get_checkpoint_path(method=method, config=config)
    trainer.save_checkpoint()

    return trainer.model, trainer.calibrator, checkpoint_path


def evaluate(method, model, calibrator, config, corruption=None, score_type="msp"):
    model.eval()
    if calibrator is not None:
        calibrator.eval()

    train_transform_id, test_transform_id = get_transforms(dataset_name=config.id_dataset)
    train_transform_ood, test_transform_ood = get_transforms(dataset_name=config.ood_dataset)
    train_data, val_data1, val_data2, test_data_id, test_data_ood = get_dataset(
        task_name=config.task_name,
        data_dir=config.data_dir,
        train_transform=train_transform_id,
        val_transform=test_transform_id,
        test_transform=test_transform_ood,
        corruption_type=corruption,
    )

    iid_test_loader = torch.utils.data.DataLoader(
        test_data_id,
        batch_size=config.test_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    ood_test_loader = torch.utils.data.DataLoader(
        test_data_ood,
        batch_size=config.test_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    if config.task == "waterbirds":
        id_ood_test_loader = None
    else:
        id_ood_test_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([test_data_id, test_data_ood]),
            batch_size=config.test_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    metrics = {}
    iid_metrics = get_measures(
        method=method,
        model=model,
        calibrator=calibrator,
        dataloader=iid_test_loader,
        dataset_name=config.id_dataset,
        thresholds=[0.9, 0.95, 0.99],
        score_type=score_type,
    )
    ood_metrics = get_measures(
        method=method,
        model=model,
        calibrator=calibrator,
        dataloader=ood_test_loader,
        dataset_name=config.ood_dataset,
        thresholds=[0.9, 0.95, 0.99],
        score_type=score_type,
    )
    id_ood_metrics = {}
    if config.task != "waterbirds":
        id_ood_metrics = get_measures(
            method=method,
            model=model,
            calibrator=calibrator,
            dataloader=id_ood_test_loader,
            dataset_name=config.id_dataset,
            thresholds=[0.9, 0.95, 0.99],
            score_type=score_type,
        )

    metrics["id"] = iid_metrics
    metrics["ood"] = ood_metrics
    metrics["id+ood"] = id_ood_metrics

    return metrics
