import argparse
from argparse import Namespace

import os
import numpy as np
import random
import torch
from mlp import MLP

from config import configs
from models import get_model
from train import train, evaluate
from utils import print_results

cifar_corruption_types = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_cifar10c_expt(config, model, calibrator, score_type):
    all_results = []
    model, calibrator, checkpoint_path = train(
        method=config.method,
        model=model,
        calibrator=calibrator,
        config=config,
        corruption="brightness",
    )  # train only on ID distribution (no use of corrupted images here)

    for corruption in cifar_corruption_types:
        results = evaluate(
            method=config.method,
            model=model,
            calibrator=calibrator,
            config=config,
            corruption=corruption,
            score_type=score_type,
        )
        all_results.append(results)
    print_results(all_results)


def run_expt(config, score_type):
    model = get_model(
        id_dataset_name=config.id_dataset,
        base_model_path=config.base_model_path,
        method=config.method,
    )
    calibrator = (
        MLP(input_size=config.num_classes, hidden_layer_size=512).cuda()
        if config.method == "bc"
        else None
    )
    if torch.cuda.is_available():
        model = model.cuda()
        if calibrator is not None:
            calibrator = calibrator.cuda()
        torch.backends.cudnn.benchmark = True

    if config.task == "cifar10":
        return run_cifar10c_expt(config, model, calibrator, score_type)

    all_results = []
    for seed in range(config.num_seeds):
        model, calibrator, checkpoint_path = train(
            method=config.method,
            model=model,
            calibrator=calibrator,
            config=config,
            corruption=None,
        )
        results = evaluate(
            method=config.method,
            model=model,
            calibrator=calibrator,
            config=config,
            corruption=None,
            score_type=score_type,
        )
        all_results.append(results)
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data-Driven Confidence Minimization Selective Classification Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", type=str, choices=["cifar10", "waterbirds", "camelyon17", "fmow"]
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dcm", "ft", "bc", "msp", "dg", "sat", "max_logit", "ensemble"],
        help="Choose a method.",
    )
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--base_model_path", nargs="+")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--confidence_weight", type=float, default=0.5)
    parser.add_argument("--repeat_uncertainty_set", type=int, default=None)
    parser.add_argument("--reward", type=float, default=2.2)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--prefetch", type=int, default=0, help="Pre-fetching threads.")
    args = parser.parse_args()
    config = configs[args.task]
    args = vars(args)
    config.update(args)
    config = Namespace(**config)

    set_seed(0)

    score_type = "msp"
    if config.method == "max_logit":
        score_type = "max_logit"

    run_expt(config, score_type)
