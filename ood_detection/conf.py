import argparse


def print_arguments(args):
    print()
    print("Arguments given for the script...")
    for key in args:
        print("Key: ", key, " Value: ", args[key])
    print()


def parse_arguments():
    ap = argparse.ArgumentParser()

    # wandb run name
    ap.add_argument("-run_name", "--run_name", type=str, default="Ours")

    # model name
    ap.add_argument("-model_type", "--model_type", type=str, default="WideResNet-40-2-0.3")

    # ID dataset
    ap.add_argument("-id_dataset_name", "--id_dataset_name", type=str, default="CIFAR10")
    ap.add_argument("-id_dataset_path", "--id_dataset_path", type=str, default="./data")
    ap.add_argument("-num_classes", "--num_classes", type=int, default=10)

    ap.add_argument(
        "-id_train_start_index",
        "--id_train_start_index",
        type=int,
        default=0,
    )
    ap.add_argument(
        "-id_train_end_index",
        "--id_train_end_index",
        type=int,
        default=4000,
    )
    ap.add_argument(
        "-id_val_start_index",
        "--id_val_start_index",
        type=int,
        default=4000,
    )
    ap.add_argument(
        "-id_val_end_index",
        "--id_val_end_index",
        type=int,
        default=5000,
    )
    ap.add_argument(
        "-id_unlabeled_start_index",
        "--id_unlabeled_start_index",
        type=int,
        default=0,
    )
    ap.add_argument(
        "-id_unlabeled_end_index",
        "--id_unlabeled_end_index",
        type=int,
        default=100,
    )
    ap.add_argument(
        "-ood_unlabeled_start_index",
        "--ood_unlabeled_start_index",
        type=int,
        default=0,
    )
    ap.add_argument(
        "-ood_unlabeled_end_index",
        "--ood_unlabeled_end_index",
        type=int,
        default=20,
    )
    ap.add_argument(
        "-id_test_start_index",
        "--id_test_start_index",
        type=int,
        default=100,
    )
    ap.add_argument(
        "-id_test_end_index",
        "--id_test_end_index",
        type=int,
        default=1000,
    )
    ap.add_argument(
        "-ood_test_start_index",
        "--ood_test_start_index",
        type=int,
        default=20,
    )
    ap.add_argument(
        "-ood_test_end_index",
        "--ood_test_end_index",
        type=int,
        default=200,
    )

    ap.add_argument(
        "-repeat_unlabeled_set",
        "--repeat_unlabeled_set",
        type=int,
        default=1,
    )

    ap.add_argument(
        "-id_dataset_start_label",
        "--id_dataset_start_label",
        type=int,
        default=0,
    )
    ap.add_argument(
        "-id_dataset_end_label",
        "--id_dataset_end_label",
        type=int,
        default=10,
    )
    ap.add_argument(
        "-ood_dataset_start_label",
        "--ood_dataset_start_label",
        type=int,
        default=0,
    )
    ap.add_argument(
        "-ood_dataset_end_label",
        "--ood_dataset_end_label",
        type=int,
        default=10,
    )
    ap.add_argument(
        "-val_is_not_train",
        "--val_is_not_train",
        action="store_false",
    )

    # OOD dataset
    ap.add_argument("-ood_dataset_name", "--ood_dataset_name", type=str, default="CIFAR100")
    ap.add_argument("-ood_dataset_path", "--ood_dataset_path", type=str, default="./data")

    # training arguments
    ap.add_argument("-confidence_weight", "--confidence_weight", type=float, default=0.5)
    ap.add_argument("-energy_weight", "--energy_weight", type=float, default=0.1)
    ap.add_argument("-train_batch_size", "--train_batch_size", type=int, default=32)
    ap.add_argument("-val_batch_size", "--val_batch_size", type=int, default=1024)
    ap.add_argument("-num_workers", "--num_workers", type=int, default=4)
    ap.add_argument("-seed", "--seed", type=int, default=0)
    ap.add_argument("-early_stop", "--early_stop", action="store_true")

    ap.add_argument("-lr", "--lr", type=float, default=0.1)
    ap.add_argument("-momentum", "--momentum", type=float, default=0.9)
    ap.add_argument("-weight_decay", "--weight_decay", type=float, default=0.0005)

    ap.add_argument("-num_epochs_train", "--num_epochs_train", type=int, default=100)
    ap.add_argument("-num_epochs_finetune", "--num_epochs_finetune", type=int, default=10)
    ap.add_argument("-use_nesterov", "--use_nesterov", type=int, default=1)
    ap.add_argument("-use_energy", "--use_energy", action="store_true")
    ap.add_argument("-verbose", "--verbose", type=bool, default=True)
    ap.add_argument(
        "-use_default_scheduler",
        "--use_default_scheduler",
        type=int,
        default=0,
        choices=[0, 1],
    )
    ap.add_argument("-should_plot", "--should_plot", type=int, default=0)

    # saving directory arguments
    ap.add_argument("-model_name", "--model_name", type=str, default="MSP model")
    ap.add_argument("-model_save_path", "--model_save_path", type=str)
    ap.add_argument("-plot_directory", "--plot_directory", type=str, default="./")
    ap.add_argument("-log_directory", "--log_directory", type=str, default="./")
    ap.add_argument(
        "-finetune_pretrained_model", "--finetune_pretrained_model", action="store_true"
    )
    ap.add_argument("-finetune_params", "--finetune_params", type=str, default="all")
    ap.add_argument("-recall_level", "--recall_level", type=float, default=0.95)
    ap.add_argument("-no_train", "--no_train", action="store_true")
    ap.add_argument("-m_in", "--m_in", type=float, default=-23.0)
    ap.add_argument("-m_out", "--m_out", type=float, default=-5.0)

    # optional test
    ap.add_argument("-run_odin", "--run_odin", action="store_true")
    ap.add_argument("-run_maha", "--run_maha", action="store_true")

    script_arguments = vars(ap.parse_args())
    return script_arguments
