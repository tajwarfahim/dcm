import torch
import numpy as np


def change_state_dict_key_names(state_dict, string_to_remove):
    for key in list(state_dict.keys()):
        state_dict[key.replace(string_to_remove, "")] = state_dict.pop(key)
    return state_dict


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    index = [item[2] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    index = torch.LongTensor(index)
    return [data, target, index]


def unpack_example(example):
    return example[0], example[1]


def get_checkpoint_path(method, config):
    if method == "dcm":
        checkpoint_path = f"{config.task_name}_{method}_confwt{config.confidence_weight}"
        if config.repeat_uncertainty_set is not None:
            checkpoint_path += f"_rep{config.repeat_uncertainty_set}"
    elif method == "dg":
        checkpoint_path = f"{config.task_name}_{method}_reward{config.reward}"
    elif method in ["ft", "sat", "bc"]:
        checkpoint_path = f"{config.task_name}_{method}"
    checkpoint_path += f"_lr{config.lr}.pt"
    return checkpoint_path


def print_results(all_results):
    for setting in ["id", "ood", "id+ood"]:
        print("Test setting: ", setting)
        acc_90 = np.mean([results[setting]["accuracy@0.9"][0] for results in all_results])
        std_acc_90 = np.mean([results[setting]["accuracy@0.9"][1] for results in all_results])
        acc_95 = np.mean([results[setting]["accuracy@0.95"][0] for results in all_results])
        std_acc_95 = np.mean([results[setting]["accuracy@0.95"][1] for results in all_results])
        acc_99 = np.mean([results[setting]["accuracy@0.99"][0] for results in all_results])
        std_acc_99 = np.mean([results[setting]["accuracy@0.99"][1] for results in all_results])
        cov_90 = np.mean([results[setting]["coverage@0.9"][0] for results in all_results])
        std_cov_90 = np.mean([results[setting]["coverage@0.9"][1] for results in all_results])
        cov_95 = np.mean([results[setting]["coverage@0.95"][0] for results in all_results])
        std_cov_95 = np.mean([results[setting]["coverage@0.95"][1] for results in all_results])
        cov_99 = np.mean([results[setting]["coverage@0.99"][0] for results in all_results])
        std_cov_99 = np.mean([results[setting]["coverage@0.99"][1] for results in all_results])
        auc = np.mean([results[setting]["auc"][0] for results in all_results])
        std_auc = np.std([results[setting]["auc"][1] for results in all_results])
        ece = np.mean([results[setting]["ece"][0] for results in all_results])
        std_ece = np.std([results[setting]["ece"][1] for results in all_results])

        print("Accuracy @ 90: {:.2f} +/- {:.2f}".format(100 * acc_90, std_acc_90))
        print("Coverage @ 90: {:.2f} +/- {:.2f}".format(100 * cov_90, std_cov_90))
        print("Accuracy @ 95: {:.2f} +/- {:.2f}".format(100 * acc_95, std_acc_95))
        print("Coverage @ 95: {:.2f} +/- {:.2f}".format(100 * cov_95, std_cov_95))
        print("Accuracy @ 99: {:.2f} +/- {:.2f}".format(100 * acc_99, std_acc_99))
        print("Coverage @ 99: {:.2f} +/- {:.2f}".format(100 * cov_99, std_cov_99))
        print("AUC: {:.2f} +/- {:.2f}".format(100 * auc, std_auc))
        print("ECE: {:.2f} +/- {:.2f}".format(ece, std_ece))
        print()
