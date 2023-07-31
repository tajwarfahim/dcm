# imports from packages
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import scipy

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

# imports from our scripts
from .other_baselines import get_ood_scores_odin
from .other_baselines import get_Mahalanobis_score
from .other_baselines import sample_estimator
from .other_baselines import fpr_and_fdr_at_recall


def process_dataset(model, dataloader, score_type="msp"):
    model.eval()
    preds = []
    targets = []
    all_scores = []

    with torch.no_grad():
        for img, label in dataloader:
            if torch.cuda.is_available():
                img = img.cuda()

            output = model(img)
            probs = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
            preds.append(np.argmax(probs, axis=1))

            if score_type == "msp":
                scores = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
                scores = np.max(a=scores, axis=1, keepdims=False)
            elif score_type == "energy":
                scores = torch.logsumexp(output, dim=1).detach().cpu().numpy()
            elif score_type == "maxlogit":
                max_elements, _ = torch.max(output, dim=1)
                scores = max_elements.detach().cpu().numpy()
            else:
                raise ValueError(f"Given score type {score_type} is not supported")

            all_scores.append(scores)
            targets.append(label.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    return preds, all_scores, targets


def get_score(model, dataloader, score_type):
    _, scores, _ = process_dataset(
        model=model,
        dataloader=dataloader,
        score_type=score_type,
    )

    assert len(scores.shape) == 1
    assert scores.shape[0] == len(dataloader.dataset)

    return scores


def get_accuracy(model, dataloader):
    preds, scores, targets = process_dataset(model=model, dataloader=dataloader)
    assert preds.shape == targets.shape
    assert len(preds.shape) == 1

    num_correct = scores[preds == targets].shape[0]

    print("Num correct: ", num_correct)
    accuracy = float(num_correct) / len(dataloader.dataset)
    return accuracy


class MSPScorer:
    def __init__(
        self,
        id_train_dataloader,
        id_test_dataloader,
        ood_test_dataloader,
        num_classes,
        id_train_dataset_mean,
    ):
        self.id_train_dataloader = id_train_dataloader
        self.id_test_dataloader = id_test_dataloader
        self.ood_test_dataloader = ood_test_dataloader
        self.num_classes = num_classes
        self.id_train_dataset_mean = id_train_dataset_mean

        print("Mean: ", self.id_train_dataset_mean)

    def calculate_msp_scores(self, model):
        self.id_msp_scores = -1.0 * get_score(
            model=model,
            dataloader=self.id_test_dataloader,
            score_type="msp",
        )
        self.ood_msp_scores = -1.0 * get_score(
            model=model,
            dataloader=self.ood_test_dataloader,
            score_type="msp",
        )

    def calculate_energy_scores(self, model):
        self.id_energy_scores = -1.0 * get_score(
            model=model,
            dataloader=self.id_test_dataloader,
            score_type="energy",
        )
        self.ood_energy_scores = -1.0 * get_score(
            model=model,
            dataloader=self.ood_test_dataloader,
            score_type="energy",
        )

    def calculate_maxlogit_scores(self, model):
        self.id_maxlogit_scores = -1.0 * get_score(
            model=model,
            dataloader=self.id_test_dataloader,
            score_type="maxlogit",
        )
        self.ood_maxlogit_scores = -1.0 * get_score(
            model=model,
            dataloader=self.ood_test_dataloader,
            score_type="maxlogit",
        )

    def calculate_odin_scores(self, model, temperature, noise):
        ood_num_examples = len(self.ood_test_dataloader.dataset)

        self.id_odin_score, _, _ = get_ood_scores_odin(
            loader=self.id_test_dataloader,
            net=model,
            bs=self.id_test_dataloader.batch_size,
            ood_num_examples=ood_num_examples,
            T=temperature,
            noise=noise,
            in_dist=True,
            mean=self.id_train_dataset_mean,
        )

        self.ood_odin_score = get_ood_scores_odin(
            loader=self.ood_test_dataloader,
            net=model,
            bs=self.ood_test_dataloader.batch_size,
            ood_num_examples=ood_num_examples,
            T=temperature,
            noise=noise,
            in_dist=False,
            mean=self.id_train_dataset_mean,
        )

    def calculate_mahalanobis_scores(self, model, noise):
        ood_num_examples = len(self.ood_test_dataloader.dataset)
        batch_size = self.ood_test_dataloader.batch_size
        num_batches = int(ood_num_examples / batch_size)

        temp_x = torch.rand(2, 3, 32, 32)
        temp_x = Variable(temp_x)
        if torch.cuda.is_available():
            temp_x = temp_x.cuda()

        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0

        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        sample_mean, precision = sample_estimator(
            model=model,
            num_classes=self.num_classes,
            feature_list=feature_list,
            train_loader=self.id_train_dataloader,
        )

        self.id_mahalanobis_score = get_Mahalanobis_score(
            model=model,
            test_loader=self.id_test_dataloader,
            num_classes=self.num_classes,
            sample_mean=sample_mean,
            precision=precision,
            layer_index=count - 1,
            magnitude=noise,
            num_batches=num_batches,
            in_dist=True,
            mean=self.id_train_dataset_mean,
        )

        self.ood_mahalanobis_score = get_Mahalanobis_score(
            model=model,
            test_loader=self.ood_test_dataloader,
            num_classes=self.num_classes,
            sample_mean=sample_mean,
            precision=precision,
            layer_index=count - 1,
            magnitude=noise,
            num_batches=num_batches,
            in_dist=False,
            mean=self.id_train_dataset_mean,
        )

    def choose_id_and_ood_scores(self, score_type):
        if score_type == "msp":
            id_scores = self.id_msp_scores
            ood_scores = self.ood_msp_scores

        elif score_type == "odin":
            id_scores = self.id_odin_score
            ood_scores = self.ood_odin_score

        elif score_type == "mahalanobis":
            id_scores = self.id_mahalanobis_score
            ood_scores = self.ood_mahalanobis_score

        elif score_type == "energy":
            id_scores = self.id_energy_scores
            ood_scores = self.ood_energy_scores

        elif score_type == "maxlogit":
            id_scores = self.id_maxlogit_scores
            ood_scores = self.ood_maxlogit_scores

        else:
            raise ValueError("Given score type is not recognized.")

        return id_scores, ood_scores

    def save_score(self, score_type, save_dir):
        id_scores, ood_scores = self.choose_id_and_ood_scores(score_type=score_type)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        id_scores_path = os.path.join(save_dir, f"{score_type}_id.txt")
        ood_scores_path = os.path.join(save_dir, f"{score_type}_ood.txt")

        np.savetxt(id_scores_path, id_scores)
        np.savetxt(ood_scores_path, ood_scores)

    def calculate_auroc(self, score_type):
        y = [1 for i in range(len(self.ood_test_dataloader.dataset))] + [
            0 for i in range(len(self.id_test_dataloader.dataset))
        ]
        y = np.array(y, dtype=np.int32)

        id_scores, ood_scores = self.choose_id_and_ood_scores(score_type=score_type)
        scores = np.append(ood_scores, id_scores, axis=0)

        auroc = roc_auc_score(y_true=y, y_score=scores)
        auroc = round(auroc * 100, 1)

        return auroc

    def calculate_aupr(self, score_type, positive):
        id_scores, ood_scores = self.choose_id_and_ood_scores(score_type=score_type)
        scores = np.append(ood_scores, id_scores, axis=0)

        if positive == "in":
            y = [0 for i in range(len(self.ood_test_dataloader.dataset))] + [
                1 for i in range(len(self.id_test_dataloader.dataset))
            ]
            scores = (-1.0) * scores
        elif positive == "out":
            y = [1 for i in range(len(self.ood_test_dataloader.dataset))] + [
                0 for i in range(len(self.id_test_dataloader.dataset))
            ]
        else:
            raise ValueError("Given value of positive dataset not recognized.")

        y = np.array(y, dtype=np.int32)
        aupr = average_precision_score(y_true=y, y_score=scores)
        aupr = round(aupr * 100, 1)

        return aupr

    def calculate_accuracy(self, model):
        accuracy = get_accuracy(model=model, dataloader=self.id_test_dataloader)
        accuracy = round(accuracy * 100, 2)

        return accuracy

    def calculate_fpr(self, score_type, recall_level):
        id_scores, ood_scores = self.choose_id_and_ood_scores(score_type=score_type)
        scores = np.append(ood_scores, id_scores, axis=0)

        y = [1 for i in range(len(self.ood_test_dataloader.dataset))] + [
            0 for i in range(len(self.id_test_dataloader.dataset))
        ]
        y = np.array(y, dtype=np.int32)

        fpr = fpr_and_fdr_at_recall(
            y_true=y, y_score=scores, recall_level=recall_level, pos_label=None
        )
        fpr = round(fpr * 100, 1)

        return fpr

    def calculate_conservative_id_accuracy(self, score_type):
        id_scores, ood_scores = self.choose_id_and_ood_scores(score_type=score_type)

        max_ood_score = np.max(ood_scores)
        num_correct = 0
        for i in range(id_scores.shape[0]):
            if id_scores[i] > max_ood_score:
                num_correct += 1

        accuracy = float(num_correct) / id_scores.shape[0]
        return accuracy

    def calculate_selective_classification_accuracy(
        self, model, threshold=0.98, score_type="msp"
    ):
        preds, all_scores, targets = process_dataset(
            model=model,
            dataloader=self.id_test_dataloader,
            score_type=score_type,
        )

        assert len(all_scores.shape) == 1
        sorted_indices = np.flip(np.argsort(all_scores))
        preds = preds[sorted_indices]
        all_scores = all_scores[sorted_indices]
        targets = targets[sorted_indices]

        cutoff_index = threshold * len(self.id_test_dataloader.dataset)
        cutoff_index = int(cutoff_index)
        pred_slice = preds[:cutoff_index]
        target_slice = targets[:cutoff_index]

        num_correct = pred_slice[pred_slice == target_slice].shape[0]
        selective_accuracy = float(num_correct) / cutoff_index

        return selective_accuracy

    def calculate_selective_classification_coverage(
        self, model, threshold=0.98, grid_size=1000, score_type="msp"
    ):
        preds, all_scores, targets = process_dataset(
            model=model,
            dataloader=self.id_test_dataloader,
            score_type=score_type,
        )

        assert len(all_scores.shape) == 1
        sorted_indices = np.flip(np.argsort(all_scores))
        preds = preds[sorted_indices]
        all_scores = all_scores[sorted_indices]
        targets = targets[sorted_indices]

        coverages = np.array([i for i in range(1, grid_size + 1)], dtype=np.float32)
        coverages /= grid_size
        indices = coverages * len(self.id_test_dataloader.dataset)

        for i in range(grid_size - 1, -1, -1):
            cutoff_index = int(indices[i])
            pred_slice = preds[:cutoff_index]
            target_slice = targets[:cutoff_index]

            num_correct = pred_slice[pred_slice == target_slice].shape[0]
            selective_accuracy = float(num_correct) / cutoff_index
            if selective_accuracy >= threshold:
                return coverages[i]

        raise ValueError(f"Selective accuracy never reaches threshold {threshold}")

    def calculate_selective_classification_auc(
        self,
        model,
        grid_size=1000,
        should_plot=False,
        figure_path=None,
        score_type="msp",
    ):
        preds, all_scores, targets = process_dataset(
            model=model,
            dataloader=self.id_test_dataloader,
            score_type=score_type,
        )

        # auc for accuracy vs coverage
        assert len(all_scores.shape) == 1
        sorted_indices = np.flip(np.argsort(all_scores))
        preds = preds[sorted_indices]
        all_scores = all_scores[sorted_indices]
        targets = targets[sorted_indices]

        coverages = np.array([i for i in range(1, grid_size + 1)], dtype=np.float32)
        coverages /= grid_size
        indices = coverages * len(self.id_test_dataloader.dataset)

        selective_accuracies = []
        for i in range(grid_size):
            cutoff_index = int(indices[i])
            pred_slice = preds[:cutoff_index]
            target_slice = targets[:cutoff_index]

            num_correct = pred_slice[pred_slice == target_slice].shape[0]
            selective_accuracy = float(num_correct) / cutoff_index
            selective_accuracies.append(selective_accuracy)

        area_under_curve = auc(x=coverages, y=selective_accuracies)

        if should_plot:
            fig = plt.figure(figsize=(5, 5))
            plt.plot(coverages, selective_accuracies, "b--", label=f"AUC: {area_under_curve}")
            plt.xlabel("Coverage (%)", fontsize="x-large")
            plt.ylabel("Selective accuracy (%)", fontsize="x-large")
            plt.title("Selective accuracy vs coverage plot", fontsize="xx-large")

            if figure_path is not None:
                plt.savefig(figure_path)

            plt.close(fig=fig)

        return area_under_curve
