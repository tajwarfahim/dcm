import numpy as np
import torch
import torch.nn.functional as F
from data.utils import unpack_example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entropy(probs: torch.Tensor, dim=-1):
    "Calcuate the entropy of a categorical probability distribution."
    log_probs = probs.log()
    ent = (-probs * log_probs).sum(dim=dim)
    return ent


def get_calibrator_output(calibrator, model_output, label):
    output_calibrator = calibrator(F.softmax(model_output, dim=-1))
    pred_calibrator = output_calibrator.argmax(dim=-1)
    mask = pred_calibrator == 0
    model_output = model_output[mask, :]
    label = label[mask]
    return model_output, label


def process_dataset(model, calibrator, dataloader, method, score_type="msp"):
    model.eval()
    preds = []
    targets = []
    all_scores = []
    all_probs = []
    abstain_probs = []
    dataloader.dataset.offset = np.random.randint(len(dataloader.dataset))

    with torch.no_grad():
        for example in dataloader:
            img, label, _ = unpack_example(example)
            img, label = img.to(device), label.to(device)
            if method in ["dg", "sat"]:
                img, label = torch.autograd.Variable(img), torch.autograd.Variable(label)

            output = model(img)
            if calibrator is not None:
                output, label = get_calibrator_output(calibrator, output, label)

            if method == "ensemble":
                probs = output.softmax(dim=-1)
                av_probs = probs.mean(dim=1)
                av_ent = entropy(probs, dim=-1).mean(dim=1)
                scores = (-1) * av_ent.detach().cpu().numpy()
                probs = av_probs.detach().cpu().numpy()
                all_probs.append(probs)
                preds.append(np.argmax(probs, axis=1))
            else:
                probs = F.softmax(output, dim=1).detach().cpu().numpy()
                if method in ["dg", "sat"]:
                    abstain_probs.extend(list((probs[:, -1])))
                    probs = probs[:, :-1]
                all_probs.append(probs)
                preds.append(np.argmax(probs, axis=1))

                if score_type == "msp":
                    scores = np.max(a=probs, axis=1, keepdims=False)
                elif score_type == "max_logit":
                    output = output.detach().cpu().numpy()
                    scores = np.max(a=output, axis=1, keepdims=False)
                elif score_type == "energy":
                    scores = torch.logsumexp(output, dim=1).detach().cpu().numpy()
                else:
                    raise ValueError(f"Given score type {score_type} is not supported")

            all_scores.append(scores)
            targets.append(label.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(preds, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(targets, axis=0)
    if method in ["dg", "sat"]:
        abstain_probs = np.array(abstain_probs)

    return all_probs, preds, all_scores, targets, abstain_probs


def confidence_loss(x):
    loss = -(x.mean(1) - torch.logsumexp(x, dim=1)).mean()
    return loss


def validate(
    method,
    model,
    calibrator,
    dataloader,
):
    model.eval()
    with torch.no_grad():
        _, preds, _, targets, _ = process_dataset(
            method=method,
            model=model,
            calibrator=calibrator,
            dataloader=dataloader,
            score_type="msp",
        )
        num_correct = (preds.squeeze() == targets.squeeze()).sum().item()
        acc = float(num_correct) / targets.shape[0]
        return acc


def train_vanilla_single_epoch(
    method,
    config,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    calibrator=None,
):
    """Fine-tunes a model using baseline approaches for one epoch."""
    model.train()
    if method == "bc":
        model.eval()
        calibrator.train()

    for i, example in enumerate(train_loader):
        data, target, index = unpack_example(example)
        if torch.cuda.is_available():
            data, target, index = data.cuda(), target.cuda(), index.cuda()
        if method in ["dg", "sat"]:
            data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

        output = model(data)
        if method == "bc":
            output = F.softmax(output, dim=-1)
            output = calibrator(output)

        if method == "dg":
            loss = criterion(outputs=output, targets=target, reward=config.reward)
        elif method == "sat":
            loss = criterion(logits=output, y=target, index=index)
        else:
            loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # model.zero_grad()

    acc = validate(method=method, model=model, calibrator=calibrator, dataloader=val_loader)

    return acc


def train_dcm_single_epoch(
    model,
    finetune_set_loader,
    uncertainty_set_loader,
    val_loader,
    optimizer,
    scheduler,
    confidence_weight,
):
    """Fine-tunes a model using Data-driven Confidence Minimization (DCM) for one epoch."""
    model.train()
    model.zero_grad()

    for in_set, out_set in zip(finetune_set_loader, uncertainty_set_loader):
        in_data = in_set[0]
        out_data = out_set[0]
        data = torch.cat((in_data, out_data), 0)
        target = in_set[1]
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        x = model(data)
        loss = F.cross_entropy(x[: len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        loss += confidence_weight * confidence_loss(x=x[len(in_set[0]) :])

        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    acc = validate(method="dcm", model=model, calibrator=None, dataloader=val_loader)

    return acc
