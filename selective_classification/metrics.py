from collections import defaultdict

import numpy as np
import sklearn.metrics as sk
from calibration.utils import get_calibration_error
from methods.utils import process_dataset

recall_level_default = 0.95
cons_level_default = 0.99


def calculate_selective_classification_accuracy(
    method, model, calibrator, dataloader, threshold=0.98, score_type="msp"
):
    _, preds, all_scores, targets, abstain_probs = process_dataset(
        model=model,
        calibrator=calibrator,
        dataloader=dataloader,
        method=method,
        score_type=score_type,
    )

    if method in ["dg", "sat"]:
        sorted_indices = np.argsort(abstain_probs)
    else:
        assert len(all_scores.shape) == 1
        sorted_indices = np.flip(np.argsort(all_scores))

    preds = preds[sorted_indices]
    targets = targets[sorted_indices]

    cutoff_index = threshold * len(dataloader.dataset)
    cutoff_index = int(cutoff_index)
    pred_slice = preds[:cutoff_index]
    target_slice = targets[:cutoff_index]

    num_correct = pred_slice[pred_slice == target_slice].shape[0]
    selective_accuracy = float(num_correct) / cutoff_index

    return selective_accuracy


def calculate_selective_classification_coverage(
    method, model, calibrator, dataloader, threshold=0.98, grid_size=1000, score_type="msp"
):
    _, preds, all_scores, targets, abstain_probs = process_dataset(
        model=model,
        calibrator=calibrator,
        dataloader=dataloader,
        method=method,
        score_type=score_type,
    )

    if method in ["dg", "sat"]:
        sorted_indices = np.argsort(abstain_probs)
    else:
        assert len(all_scores.shape) == 1
        sorted_indices = np.flip(np.argsort(all_scores))

    preds = preds[sorted_indices]
    all_scores = all_scores[sorted_indices]
    targets = targets[sorted_indices]

    coverages = np.array([i for i in range(1, grid_size + 1)], dtype=np.float32)
    coverages /= grid_size
    indices = coverages * len(dataloader.dataset)

    for i in range(grid_size - 1, -1, -1):
        cutoff_index = int(indices[i])
        pred_slice = preds[:cutoff_index]
        target_slice = targets[:cutoff_index]

        num_correct = pred_slice[pred_slice == target_slice].shape[0]
        selective_accuracy = float(num_correct) / cutoff_index
        if selective_accuracy >= threshold:
            return coverages[i]

    print(f"Selective accuracy never reaches threshold {threshold}")
    return 0


def calculate_selective_classification_auc(
    method, model, calibrator, dataloader, grid_size=1000, score_type="msp"
):
    _, preds, all_scores, targets, abstain_probs = process_dataset(
        model=model,
        calibrator=calibrator,
        dataloader=dataloader,
        method=method,
        score_type=score_type,
    )

    # auc for accuracy vs coverage
    assert len(all_scores.shape) == 1
    if method in ["dg", "sat"]:
        sorted_indices = np.argsort(abstain_probs)
    else:
        sorted_indices = np.flip(np.argsort(all_scores))
    preds = preds[sorted_indices]
    all_scores = all_scores[sorted_indices]
    targets = targets[sorted_indices]

    coverages = np.array([i for i in range(1, grid_size + 1)], dtype=np.float32)
    coverages /= grid_size
    indices = coverages * len(dataloader.dataset)

    selective_accuracies = []
    for i in range(grid_size):
        cutoff_index = int(indices[i])
        pred_slice = preds[:cutoff_index]
        target_slice = targets[:cutoff_index]

        num_correct = pred_slice[pred_slice == target_slice].shape[0]
        selective_accuracy = float(num_correct) / cutoff_index
        selective_accuracies.append(selective_accuracy)

    area_under_curve = sk.auc(x=coverages, y=selective_accuracies)

    return area_under_curve


def get_metrics(method, model, calibrator, dataloader, score_type="msp"):
    probs, preds, scores, targets, _ = process_dataset(
        model=model,
        calibrator=calibrator,
        dataloader=dataloader,
        method=method,
        score_type=score_type,
    )
    assert preds.shape == targets.shape
    ece = 100 * get_calibration_error(probs=probs, labels=targets)

    return preds, scores, targets, ece


def get_measures(
    method,
    model,
    calibrator,
    dataloader,
    dataset_name,
    thresholds=[0.9, 0.95, 0.99],
    score_type="msp",
):
    accs = defaultdict(list)
    covs = defaultdict(list)
    aucs = []
    eces = []
    grid_size = 100 if "waterbirds" in dataset_name else 1000
    for threshold in thresholds:
        accs[threshold].append(
            calculate_selective_classification_accuracy(
                method=method,
                model=model,
                calibrator=calibrator,
                dataloader=dataloader,
                threshold=threshold,
                score_type=score_type,
            )
        )
        covs[threshold].append(
            calculate_selective_classification_coverage(
                method=method,
                model=model,
                calibrator=calibrator,
                dataloader=dataloader,
                threshold=threshold,
                score_type=score_type,
            )
        )
    auc = calculate_selective_classification_auc(
        method=method,
        model=model,
        calibrator=calibrator,
        dataloader=dataloader,
        grid_size=grid_size,
        score_type=score_type,
    )
    preds, scores, targets, ece = get_metrics(
        method=method,
        model=model,
        calibrator=calibrator,
        dataloader=dataloader,
        score_type=score_type,
    )
    aucs.append(auc)
    eces.append(ece)

    metrics = {}
    for metric in ["accuracy", "coverage", "auc", "ece"]:
        for threshold in thresholds:
            metrics[f"accuracy@{threshold}"] = (
                np.mean(accs[threshold]),
                np.std(accs[threshold]),
            )
            metrics[f"coverage@{threshold}"] = (
                np.mean(covs[threshold]),
                np.std(covs[threshold]),
            )
        if metric == "auc":
            metrics[metric] = (np.mean(aucs), np.std(aucs))
        elif metric == "ece":
            metrics[metric] = (np.mean(eces), np.std(eces))

    return metrics
