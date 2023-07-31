import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from utils import change_state_dict_key_names


def get_model(id_dataset_name, base_model_path, method):
    if method == "ensemble":
        models = []
        for model_path in base_model_path:
            models.append(get_base_model(id_dataset_name, model_path, method))
        net = ModelEnsemble(models)
    else:
        net = get_base_model(id_dataset_name, base_model_path[0], method)

    return net


def get_base_model(id_dataset_name, base_model_path, method):
    if id_dataset_name == "cifar10":
        net = load_model(
            model_name="Standard",
            model_dir="/iris/u/cchoi1/exploring_ood/checkpoints",
            dataset=id_dataset_name,
            threat_model=ThreatModel.corruptions,
        )
        if method in ["dg", "sat"]:
            net.fc = torch.nn.Linear(in_features=640, out_features=11, bias=True)
            net.load_state_dict(torch.load(base_model_path))

    elif "waterbirds" in id_dataset_name:
        net = models.resnet50(pretrained=True)
        d_features = net.fc.in_features
        if method in ["dg", "sat"]:
            net.fc = nn.Linear(d_features, 3)
        else:
            net.fc = nn.Linear(d_features, 2)
        checkpoint = torch.load(base_model_path)
        net.load_state_dict(checkpoint)

    elif "camelyon17" in id_dataset_name:
        net = models.densenet121(pretrained=True)
        d_features = net.classifier.in_features
        if method in ["dg", "sat"]:
            net.classifier = nn.Linear(d_features, 3)
        else:
            net.classifier = nn.Linear(d_features, 2)
        checkpoint = torch.load(base_model_path)
        checkpoint = change_state_dict_key_names(
            state_dict=checkpoint["algorithm"], string_to_remove="model."
        )
        net.load_state_dict(checkpoint)

    elif "fmow" in id_dataset_name:
        net = models.densenet121(pretrained=True)
        d_features = net.classifier.in_features
        if method in ["dg", "sat"]:
            net.classifier = nn.Linear(d_features, 63)
        else:
            net.classifier = nn.Linear(d_features, 62)
        checkpoint = torch.load(base_model_path)
        checkpoint = change_state_dict_key_names(
            state_dict=checkpoint["algorithm"], string_to_remove="model."
        )
        net.load_state_dict(checkpoint)

    return net


class ModelEnsemble(nn.Module):
    def __init__(self, models):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList()
        for model in models:
            self.models.append(model)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs, dim=1)


class MLP(nn.Module):
    """
    MLP for binary classifier.
    """

    def __init__(self, input_size, hidden_layer_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
