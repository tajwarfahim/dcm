from __future__ import print_function
import torch
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import random

# import from our scripts
from dataloading_utils import create_dataloaders

from utils.wide_resnet_pytorch import create_wide_resnet
from utils.resnet_pytorch import ResNet18, ResNet34, ResNet50
from utils.resnet_small_pytorch import resnet20

from utils.pytorch_classifier_trainer import ClassifierTrainer

from utils.siamese_network import process_shared_model_name_wide_resnet

from utils.plotting_log_utils import plot_loss
from utils.plotting_log_utils import plot_accuracy

from utils.get_readable_timestamp import get_readable_timestamp
from conf import parse_arguments, print_arguments


def set_seed_everywhere(seed):
    print(f"Setting seed to: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_checkpoint_name(args):
    model_path = Path(
        str(f"id_{args['id_dataset_name']}_model_{args['model_type']}_seed_{args['seed']}.pt")
    )
    return model_path


def create_resnet_given_params(model_name, num_classes, num_input_channels):
    resnet_model = None

    if model_name == "ResNet18":
        resnet_model = ResNet18(
            contains_last_layer=True,
            num_input_channels=num_input_channels,
            num_classes=num_classes,
        )

    elif model_name == "ResNet20":
        resnet_model = resnet20(
            num_input_channels=num_input_channels,
            num_classes=num_classes,
        )

    elif model_name == "ResNet34":
        resnet_model = ResNet34(
            contains_last_layer=True,
            num_input_channels=num_input_channels,
            num_classes=num_classes,
        )

    elif model_name == "ResNet50":
        resnet_model = ResNet50(
            contains_last_layer=True,
            num_input_channels=num_input_channels,
            num_classes=num_classes,
        )

    else:
        raise ValueError("Model name not supported")

    return resnet_model


def create_wide_resnet_model(args):
    architecture_map = process_shared_model_name_wide_resnet(
        shared_model_name=args["model_type"]
    )
    dataset_name = args["id_dataset_name"]

    kwargs = {
        "depth": architecture_map["depth"],
        "widen_factor": architecture_map["widen_factor"],
        "dropRate": architecture_map["dropRate"],
        "num_classes": args["num_classes"],
        "contains_last_layer": True,
    }

    if (
        dataset_name == "CIFAR10"
        or dataset_name == "CIFAR100"
        or dataset_name == "CIFAR"
        or dataset_name == "SVHN"
        or dataset_name == "CIFAR100Coarse"
        or dataset_name == "TinyImageNet"
    ):
        kwargs["num_input_channels"] = 3
        model = create_wide_resnet(**kwargs)

    elif dataset_name == "MNIST":
        kwargs["num_input_channels"] = 1
        model = create_wide_resnet(**kwargs)

    else:
        raise ValueError("Dataset name not supported")

    return model


def create_resnet_model(args):
    dataset_name = args["id_dataset_name"]
    kwargs = {"model_name": args["model_type"], "num_classes": args["num_classes"]}

    if (
        dataset_name == "CIFAR10"
        or dataset_name == "CIFAR100"
        or dataset_name == "CIFAR"
        or dataset_name == "SVHN"
        or dataset_name == "CIFAR100Coarse"
        or dataset_name == "TinyImageNet"
    ):
        kwargs["num_input_channels"] = 3
        model = create_resnet_given_params(**kwargs)

    elif dataset_name == "MNIST":
        kwargs["num_input_channels"] = 1
        model = create_resnet_given_params(**kwargs)

    else:
        raise ValueError("Dataset name not supported")

    return model


def create_model(args):
    if args["model_type"].find("WideResNet") == 0:
        model = create_wide_resnet_model(args)

    elif args["model_type"].find("ResNet") >= 0:
        model = create_resnet_model(args)

    else:
        raise ValueError("Given model type is not supported")

    if torch.cuda.is_available():
        model.cuda()

    return model


def create_optimizer_and_scheduler(args, model, len_train_dataloader):
    num_epochs = (
        args["num_epochs_train"]
        if not args["finetune_pretrained_model"]
        else args["num_epochs_finetune"]
    )

    use_nesterov = None
    if args["use_nesterov"] == 1:
        use_nesterov = True
    elif args["use_nesterov"] == 0:
        use_nesterov = False
    else:
        raise ValueError("argument for using nesterov momentum, is not supported")

    if args["finetune_params"] == "all":
        params = model.parameters()

    elif args["finetune_params"] == "last":
        params = model.fc.parameters()

    elif args["finetune_params"] == "first":
        params = model.conv1.parameters()

    else:
        raise ValueError("Given finetune_params is not supported")

    optimizer = torch.optim.SGD(
        params,
        lr=args["lr"],
        momentum=args["momentum"],
        nesterov=use_nesterov,
        weight_decay=args["weight_decay"],
    )

    if args["use_default_scheduler"] == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
        )

    else:

        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

        # scheduler from outlier exposure
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                num_epochs * len_train_dataloader,
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args["lr"],
            ),
        )

    return optimizer, scheduler


def run_model_training(args):
    set_seed_everywhere(seed=args["seed"])
    model = create_model(args=args)
    if args["verbose"]:
        print("Model created.")

    model_path = None
    model_name = args["model_name"]
    if args["finetune_pretrained_model"]:
        assert os.path.isfile(args["model_save_path"])
        model.load_state_dict(torch.load(args["model_save_path"]))
        if args["verbose"]:
            print("\nPretrained weights loaded from: ", args["model_save_path"], "\n")
    else:
        model_path = os.path.join(args["model_save_path"], "Pretrained_Models")
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, get_checkpoint_name(args))

    if args["no_train"]:
        return model

    dataloaders = create_dataloaders(args=args)
    train_dataloader = dataloaders["train"]
    unlabeled_loader = dataloaders["unlabeled"]
    val_dataloader = dataloaders["val"]

    print("\nNum train batches: ", len(train_dataloader))
    print("Num unlabeled batches: ", len(unlabeled_loader))
    print("Num validation batches: ", len(val_dataloader), "\n")

    optimizer, scheduler = create_optimizer_and_scheduler(
        args=args, model=model, len_train_dataloader=len(train_dataloader)
    )
    if args["verbose"]:
        print("Optimizer and scheduler created")
        print("Optimizer: ", optimizer)
        print("scheduler: ", scheduler, "\n")

    if args["verbose"]:
        print("Model path: ", model_path, "\n")

    use_unlabeled = args["finetune_pretrained_model"] or args["use_energy"]
    if not use_unlabeled:
        unlabeled_loader = train_dataloader

    model_trainer = ClassifierTrainer(
        model=model,
        model_name=model_name,
        train_loader=train_dataloader,
        unlabeled_loader=unlabeled_loader,
        validation_loader=val_dataloader,
        confidence_weight=args["confidence_weight"],
        energy_weight=args["energy_weight"],
        m_in=args["m_in"],
        m_out=args["m_out"],
        optimizer=optimizer,
        scheduler=scheduler,
        use_unlabeled=args["finetune_pretrained_model"],
        use_energy=args["use_energy"],
    )

    # should_use_scheduler -> True, use scheduler at every batch (instead of every epoch) like Outlier-exposure paper
    # should_use_scheduler -> False, use scheduler at every epoch
    if args["use_default_scheduler"] == 1:
        should_use_scheduler = False
    else:
        should_use_scheduler = True

    num_epochs = args["num_epochs_train"]
    if args["finetune_pretrained_model"]:
        num_epochs = args["num_epochs_finetune"]

    model_trainer.run_training(
        num_epochs=num_epochs,
        model_path=model_path,
        verbose=args["verbose"],
        should_use_scheduler=should_use_scheduler,
    )

    model_trainer.report_peak_performance()

    plot_directory = os.path.join(args["plot_directory"], get_checkpoint_name(args))
    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    if args["should_plot"] > 0:
        if args["verbose"]:
            print("Plot directory: ", plot_directory)
            print("Plotting loss and accuracy...")
            print()

        plot_loss(
            model_name,
            model_trainer.train_loss_history,
            model_trainer.val_loss_history,
            plot_directory,
        )
        plot_accuracy(
            model_name,
            model_trainer.train_accuracy_history,
            model_trainer.val_accuracy_history,
            plot_directory,
        )

    log_directory = os.path.join(args["log_directory"], get_checkpoint_name(args))
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)
    if args["verbose"]:
        print("Log directory: ", log_directory)
        print("Saving training log...")
        print()

    model_trainer.save_log(log_directory)
    model = model_trainer.model
    if args["early_stop"]:
        model.load_state_dict(model_trainer.best_weights)

    return model_trainer.model


if __name__ == "__main__":
    script_arguments = parse_arguments()
    print_arguments(args=script_arguments)
    run_model_training(args=script_arguments)
