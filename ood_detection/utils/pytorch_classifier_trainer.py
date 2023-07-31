# Citation:
# 1. https://thenewstack.io/tutorial-train-a-deep-learning-model-in-pytorch-and-export-it-to-onnx/
# 2. https://discuss.pytorch.org/t/creating-custom-dataset-from-inbuilt-pytorch-datasets-along-with-data-transformations/58270/2
# 3. https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54
# 4. https://github.com/fangpin/siamese-pytorch/blob/master/train.py
# 5. https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

import torch
import torch.nn.functional as F
import time
import numpy as np
import os
import copy


def print_message(message, verbose=True):
    if verbose:
        print("")
        print(message)
        print("")


def confidence_loss(x):
    loss = -(x.mean(1) - torch.logsumexp(x, dim=1)).mean()
    return loss


def energy_loss(output, img, m_in, m_out):
    """

    Parameters
    ----------
    output
    img
    m_in: margin for in-distribution; above this value will be penalized
    m_out: margin for out-distribution; below this value will be penalized

    Returns: energy_loss
    -------

    """
    Ec_out = -torch.logsumexp(output[len(img[0]) :], dim=1)
    Ec_in = -torch.logsumexp(output[: len(img[0])], dim=1)
    energy_loss = (
        torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean()
    )

    return energy_loss


# should_use_scheduler -> True, use scheduler at every batch (instead of every epoch) like Outlier-exposure paper
# should_use_scheduler -> False, use scheduler at every epoch
def train_single_epoch(
    model,
    train_loader,
    unlabeled_loader,
    optimizer,
    scheduler,
    confidence_weight,
    energy_weight,
    use_unlabeled,
    use_energy,
    m_in,
    m_out,
):
    model.train()
    model.zero_grad()

    num_batches = 0
    for in_set, out_set in zip(train_loader, unlabeled_loader):
        num_batches += 1
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # forward
        x = model(data)

        loss = F.cross_entropy(x[: len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        if use_unlabeled:
            if use_energy:
                Ec_out = -torch.logsumexp(x[len(in_set[0]) :], dim=1)
                # Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
                # loss += energy_weight * (torch.pow(F.relu(Ec_in-m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean())
                loss += energy_weight * torch.pow(F.relu(m_out - Ec_out), 2).mean()
            else:
                loss += confidence_weight * confidence_loss(x=x[len(in_set[0]) :])

        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    print("Num batches: ", num_batches)


def test_model(model, test_loader):
    model.eval()
    xent = torch.nn.CrossEntropyLoss(reduction="sum")

    confidence = 0
    classification_loss = 0
    num_correct = 0
    num_labeled = 0

    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            output = model.forward(img).squeeze()
            loss = xent(output, label)
            classification_loss += loss.item()

            pred = torch.argmax(input=output, dim=1)
            num_correct += (pred.squeeze() == label.squeeze()).float().sum().item()
            num_labeled += pred.shape[0]

            confidence += confidence_loss(output)

    accuracy = float(num_correct) / num_labeled
    average_classification_loss = float(classification_loss) / num_labeled
    average_confidence = float(confidence) / num_labeled
    return average_classification_loss, average_confidence, accuracy


class ClassifierTrainer:
    def __init__(
        self,
        model,
        model_name,
        train_loader,
        unlabeled_loader,
        validation_loader,
        confidence_weight,
        energy_weight,
        m_in,
        m_out,
        optimizer,
        scheduler=None,
        use_unlabeled=False,
        use_energy=False,
    ):
        if use_energy:
            use_unlabeled = True

        self.model = model
        self.model_name = model_name

        self.train_loader = train_loader
        self.unlabeled_loader = unlabeled_loader
        self.validation_loader = validation_loader

        self.use_unlabeled = use_unlabeled
        self.use_energy = use_energy
        self.confidence_weight = confidence_weight
        self.energy_weight = energy_weight

        self.m_in = m_in
        self.m_out = m_out

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.max_val_accuracy = None
        self.best_epoch = None

        self.train_loss_history = []
        self.train_accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.time_history = []

        print("Using an unlabeled set to minimize confidence: ", self.use_unlabeled)
        print("Using energy based fine-tuning: ", self.use_energy)

    def run_training(
        self, num_epochs, model_path=None, verbose=True, should_use_scheduler=False
    ):

        print_message(message="Model training is starting...", verbose=verbose)

        for epoch in range(1, num_epochs + 1):

            start_time = time.time()

            # train on labeled ID + unlabeled ID/OOD data
            train_single_epoch(
                model=self.model,
                train_loader=self.train_loader,
                unlabeled_loader=self.unlabeled_loader,
                optimizer=self.optimizer,
                confidence_weight=self.confidence_weight,
                energy_weight=self.energy_weight,
                scheduler=self.scheduler,
                use_unlabeled=self.use_unlabeled,
                use_energy=self.use_energy,
                m_in=self.m_in,
                m_out=self.m_out,
            )

            # calculate the training and validation loss and accuracy
            train_classification_loss, train_confidence, train_accuracy = test_model(
                model=self.model, test_loader=self.train_loader
            )
            val_classification_loss, val_confidence, val_accuracy = test_model(
                model=self.model, test_loader=self.validation_loader
            )

            end_time = time.time()

            if not should_use_scheduler and self.scheduler is not None:
                self.scheduler.step()

            self.train_loss_history.append(train_classification_loss)
            self.train_accuracy_history.append(train_accuracy)
            self.val_loss_history.append(val_classification_loss)
            self.val_accuracy_history.append(val_accuracy)

            time_per_epoch = end_time - start_time
            self.time_history.append(time_per_epoch)

            if verbose:
                print(
                    "Epoch: ",
                    epoch,
                    "/",
                    num_epochs,
                    " Train Accuracy: ",
                    train_accuracy,
                    " Val Accuracy: ",
                    val_accuracy,
                )
                print(
                    "            Train classification loss: ",
                    train_classification_loss,
                    " Val classification loss: ",
                    val_classification_loss,
                )
                print(
                    "            Train confidence: ",
                    train_confidence,
                    " Val confidence: ",
                    val_confidence,
                )
                print("Learning rate: ", self.scheduler.get_last_lr()[0])
                print()

            if epoch == 1 or val_accuracy > self.max_val_accuracy:
                self.best_epoch = epoch
                self.max_val_accuracy = val_accuracy
                self.best_weights = copy.deepcopy(self.model.state_dict())

        if model_path is not None:
            torch.save(self.model.state_dict(), model_path)
            print("\n Saving model weights to: ", model_path)

        if verbose:
            total_time = 0
            for time_epoch in self.time_history:
                total_time += time_epoch

            print()
            print("Total training time: ", total_time, " seconds")
            print()

    def report_peak_performance(self):
        if self.max_val_accuracy == None:
            print("Model has not been trained yet.")
        else:
            print()
            print("Model peaked in validation accuracy at epoch ", self.best_epoch)
            print("Model peak validation accuracy: ", self.max_val_accuracy)
            print()

    def save_log(self, log_dir):
        print_message("Log directory: " + log_dir, verbose=1)

        record_list = [
            self.train_loss_history,
            self.val_loss_history,
            self.train_accuracy_history,
            self.val_accuracy_history,
        ]
        filename_list = ["train_loss", "val_loss", "acc", "val_acc"]
        filename_prefix = "_per_epoch.txt"

        for i in range(len(record_list)):
            numpy_record = np.array(record_list[i])
            filename = filename_list[i] + filename_prefix
            filepath = os.path.join(log_dir, filename)
            np.savetxt(filepath, numpy_record, delimiter=",", fmt="%1.4e")

        timer_log = np.array(self.time_history)
        total_time = np.sum(timer_log) / 3600.0

        appended = [total_time] + self.time_history
        appended_log = np.array(appended)
        timer_path = os.path.join(log_dir, "training_time.txt")
        np.savetxt(timer_path, appended_log, delimiter=",", fmt="%1.4e")
