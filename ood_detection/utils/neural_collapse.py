from collections import defaultdict
from torch.distributions import Categorical

import torch.nn.functional as F

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import torch


@torch.no_grad()
def get_activations_and_logits(model, test_loader):
    def get_input(name):
        return lambda model, _input, output: info_dict[name].append(_input[0].detach().cpu())

    info_dict = defaultdict(list)
    hooks = [
        model.layer2[-1].register_forward_hook(get_input("layer2")),
        model.layer3[-1].register_forward_hook(get_input("layer3")),
        model.layer4[-1].register_forward_hook(get_input("layer4")),
        model.linear.register_forward_hook(get_input("linear")),
    ]

    for img, label in test_loader:
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        output = model.forward(img).squeeze()
        info_dict["logits"].append(output.detach().cpu())
        info_dict["labels"].append(label.detach().cpu())

    for hook in hooks:
        hook.remove()
    for key in info_dict:
        info_dict[key] = torch.cat(info_dict[key], dim=0)
    info_dict["probs"] = info_dict["logits"].softmax(dim=1)
    return info_dict


def get_neural_collapse_metrics(model, dataloaders):
    info = dict()
    info["id_train"] = get_activations_and_logits(model, dataloaders["id_train_dataloader"])
    info["id_test"] = get_activations_and_logits(model, dataloaders["id_test_dataloader"])
    for ood_dataset_name in dataloaders["ood_dataloaders"]:
        info[ood_dataset_name] = get_activations_and_logits(
            model, dataloaders["ood_dataloaders"][ood_dataset_name]
        )

    output_dict = dict()
    for name, _info in info.items():
        max_probs = _info["probs"].max(dim=1)[0]
        sns.kdeplot(max_probs, label=name)
        output_dict[f"{name}/msp_mean"] = max_probs.mean()
    plt.legend()
    output_dict["fig/msp"] = wandb.Image(plt)
    plt.cla()

    for name, _info in info.items():
        entropy = Categorical(_info["probs"]).entropy()
        sns.kdeplot(entropy, label=name)
        output_dict[f"{name}/entropy_mean"] = entropy.mean()
    plt.legend()
    output_dict["fig/entropy"] = wandb.Image(plt)
    plt.cla()

    tr_info = info["id_train"]
    C = tr_info["labels"].unique().shape[0]
    global_mean = tr_info["linear"].mean(dim=0)
    class_means = [
        tr_info["linear"][tr_info["labels"] == y].mean(dim=0)
        for y in tr_info["labels"].unique()
    ]
    mean_dists = [(c - global_mean).norm(dim=0) for c in class_means]
    mean_dists = torch.stack(mean_dists)
    output_dict[f"id_train/coeff_var_L2_c-g"] = (mean_dists.std() / mean_dists.mean()).item()

    norm_class_means = F.normalize(torch.stack(class_means))
    cosines = torch.mm(norm_class_means, norm_class_means.T)
    cosines = torch.triu(cosines, diagonal=1)
    cosines_flat = cosines[cosines.nonzero(as_tuple=True)]
    output_dict[f"id_train/cosine_std"] = cosines_flat.std().item()
    output_dict[f"id_train/cosine_mean"] = cosines_flat.mean().item() + 1 / (C - 1)

    for name, _info in info.items():
        activations = _info["linear"].unsqueeze(0)
        class_mean_stacked = torch.stack(class_means).unsqueeze(1)
        dists = (activations - class_mean_stacked).norm(dim=-1)
        min_dists = dists.min(dim=0)[0]
        sns.kdeplot(min_dists, label=name)
        output_dict[f"{name}/euclidean_class_mean"] = min_dists.mean()
    plt.legend()
    output_dict["fig/euclidean_class_mean"] = wandb.Image(plt)
    plt.cla()

    output_dict = {f"collapse/" + k: v for k, v in output_dict.items()}
    return output_dict
