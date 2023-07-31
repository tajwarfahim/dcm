import torch


class BaseAlgorithm:
    def __init__(
        self,
        config,
        model,
        calibrator,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        setting,
        checkpoint_path,
    ):
        self.config = config
        self.model = model
        self.calibrator = calibrator
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.setting = setting
        self.checkpoint_path = checkpoint_path

    def run_training(self, num_epochs):
        pass

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)
