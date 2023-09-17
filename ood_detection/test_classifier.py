from __future__ import print_function

import os
import time
import warnings
import wandb


# imports from our scripts
from utils.msp_scorer import MSPScorer
from conf import parse_arguments, print_arguments
from dataloading_utils import create_dataloaders
from train_classifier import run_model_training


def integrate_wandb(args):
    id_dataset_name = args["id_dataset_name"]
    ood_dataset_name = args["ood_dataset_name"]
    project_name = f"OOD_detection_{id_dataset_name}_vs_{ood_dataset_name}"
    run = wandb.init(
        name=args["run_name"],
        project=project_name,
    )
    wandb.config.update(args)


def run_odin_experiment(args, scorer, model):
    # search grid for noise value used in mahalanobis paper's code
    noise_values = [0.0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]

    noise_values = list(set(noise_values))
    noise_values.sort()

    # search grid for temperature value used in mahalanobis paper's code
    temperature_values = [1000.0] #[1.0, 10.0, 100.0, 1000.0]

    print("Running ODIN baseline.")
    print("Temperature values we try: ", temperature_values)
    print("Noise magnitude values we try: ", noise_values)

    best_cons_metric = None
    default_temperature = 1000.0
    default_noise = 0.0

    for temperature in temperature_values:
        for noise in noise_values:
            # we are in the MSP case, so we ignore this
            if temperature == 1.0 and noise == 0.0:
                continue

            scorer.calculate_odin_scores(model=model, temperature=temperature, noise=noise)
            auroc = scorer.calculate_auroc(score_type="odin")
            aupr_in = scorer.calculate_aupr(score_type="odin", positive="in")
            aupr_out = scorer.calculate_aupr(score_type="odin", positive="out")
            fpr = scorer.calculate_fpr(score_type="odin", recall_level=args["recall_level"])
            cons_metric = scorer.calculate_fpr(score_type="odin", recall_level=0.99)

            if temperature == default_temperature and noise == default_noise:
                default_auroc = auroc
                default_aupr_in = aupr_in
                default_aupr_out = aupr_out
                default_fpr = fpr
                default_cons_metric = cons_metric

            if best_cons_metric is None or cons_metric < best_cons_metric:
                best_noise_value = noise
                best_temperature_value = temperature
                best_auroc = auroc
                best_aupr_in = aupr_in
                best_aupr_out = aupr_out
                best_fpr = fpr
                best_cons_metric = cons_metric

    wandb.log(
        {
            "odin_auroc": best_auroc,
            "odin_FPR_95": best_fpr,
            "odin_FPR_99": best_cons_metric,
            "odin_aupr_in": best_aupr_in,
            "odin_aupr_out": best_aupr_out,
        },
        step=0,
    )


def run_mahalanobis_experiment(scorer, args, model):
    noise_values = [0.0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]

    best_cons_metric = None
    default_noise = 0.0

    for noise in noise_values:
        scorer.calculate_mahalanobis_scores(model=model, noise=noise)

        auroc = scorer.calculate_auroc(score_type="mahalanobis")
        aupr_in = scorer.calculate_aupr(score_type="mahalanobis", positive="in")
        aupr_out = scorer.calculate_aupr(score_type="mahalanobis", positive="out")
        fpr = scorer.calculate_fpr(score_type="mahalanobis", recall_level=args["recall_level"])
        cons_metric = scorer.calculate_fpr(score_type="mahalanobis", recall_level=0.99)

        if noise == default_noise:
            default_auroc = auroc
            default_aupr_in = aupr_in
            default_aupr_out = aupr_out
            default_fpr = fpr
            default_cons_metric = cons_metric

        if best_cons_metric is None or cons_metric < best_cons_metric:
            best_noise_value = noise
            best_auroc = auroc
            best_aupr_in = aupr_in
            best_aupr_out = aupr_out
            best_fpr = fpr
            best_cons_metric = cons_metric

    wandb.log(
        {
            "maha_auroc": best_auroc,
            "maha_FPR_95": best_fpr,
            "maha_FPR_99": best_cons_metric,
            "maha_aupr_in": best_aupr_in,
            "maha_aupr_out": best_aupr_out,
        },
        step=0,
    )


def run_no_extra_hyperparam_experiment(scorer, args, model, score_type):
    score_path = os.path.join(
        "scores",
        f"ID_{args['id_dataset_name']}_OOD_{args['ood_dataset_name']}",
        args["run_name"],
        str(args["seed"]),
    )

    if score_type == "msp":
        scorer.calculate_msp_scores(model=model)
    elif score_type == "energy":
        scorer.calculate_energy_scores(model=model)
    elif score_type == "maxlogit":
        scorer.calculate_maxlogit_scores(model=model)
    else:
        raise ValueError("Given score type not supported.")

    scorer.save_score(score_type=score_type, save_dir=score_path)

    auroc = scorer.calculate_auroc(score_type=score_type)
    aupr_in = scorer.calculate_aupr(score_type=score_type, positive="in")
    aupr_out = scorer.calculate_aupr(score_type=score_type, positive="out")
    fpr = scorer.calculate_fpr(score_type=score_type, recall_level=args["recall_level"])
    cons_metric = scorer.calculate_fpr(score_type=score_type, recall_level=0.99)

    selective_classification_auc = scorer.calculate_selective_classification_auc(
        model=model,
        should_plot=args["should_plot"],
        figure_path=os.path.join("./figures", f"selective_auc_figure.jpg"),
        score_type=score_type,
    )

    selective_accuracy = scorer.calculate_selective_classification_accuracy(
        model=model, threshold=0.98, score_type=score_type
    )
    coverage = scorer.calculate_selective_classification_coverage(
        model=model, threshold=0.98, score_type=score_type
    )

    wandb.log(
        {
            f"{score_type}_auroc": auroc,
            f"{score_type}_FPR_95": fpr,
            f"{score_type}_FPR_99": cons_metric,
            f"{score_type}_aupr_in": aupr_in,
            f"{score_type}_aupr_out": aupr_out,
            f"{score_type}_selective_classification_auc": selective_classification_auc,
            f"{score_type}_selective_accuracy": selective_accuracy,
            f"{score_type}_coverage": coverage,
        },
        step=0,
    )

def run_script(args):
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    integrate_wandb(args=args)

    start_time = time.time()
    if args["should_plot"] and not os.path.isdir("./figures"):
        os.makedirs("./figures")

    model = run_model_training(args=args)
    dataloaders = create_dataloaders(args=args)
    id_train_dataloader = dataloaders["train"]
    id_test_dataloader = dataloaders["test"]
    ood_test_dataloader = dataloaders["ood"]
    mean = dataloaders["mean"]
    std = dataloaders["std"]

    print("\nID train dataset mean: ", mean)
    print("ID train dataset std: ", std, "\n")

    scorer = MSPScorer(
        id_train_dataloader=id_train_dataloader,
        id_test_dataloader=id_test_dataloader,
        ood_test_dataloader=ood_test_dataloader,
        num_classes=args["num_classes"],
        id_train_dataset_std=std,
    )

    accuracy = scorer.calculate_accuracy(model=model)
    wandb.log({"ID test accuracy": accuracy}, step=0)

    for score_type in ["msp", "energy", "maxlogit"]:
        run_no_extra_hyperparam_experiment(
            scorer=scorer,
            model=model,
            args=args,
            score_type=score_type,
        )

    if args["run_odin"]:
        run_odin_experiment(args=args, scorer=scorer, model=model)
    if args["run_maha"]:
        run_mahalanobis_experiment(scorer=scorer, args=args, model=model)

    end_time = time.time()

    total_time = end_time - start_time
    print()
    print("Total time to run this experiment: ", total_time, " s")
    print()


if __name__ == "__main__":
    script_arguments = parse_arguments()
    print_arguments(args=script_arguments)
    run_script(args=script_arguments)
