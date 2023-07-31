from .base_trainer import BaseAlgorithm
from .utils import train_vanilla_single_epoch


class FT(BaseAlgorithm):
    def __init__(
        self,
        config,
        model,
        calibrator,
        finetuning_set_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        setting,
        checkpoint_path,
    ):
        BaseAlgorithm.__init__(
            self,
            config=config,
            model=model,
            calibrator=calibrator,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            setting=setting,
            checkpoint_path=checkpoint_path,
        )
        self.finetuning_set_loader = finetuning_set_loader
        print("Fine-Tuning")

    def run_training(self, num_epochs):
        for epoch in range(num_epochs):
            val_acc = train_vanilla_single_epoch(
                method="ft",
                config=self.config,
                model=self.model,
                train_loader=self.finetuning_set_loader,
                val_loader=self.val_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            print("Epoch {}, Validation Accuracy: {:.2f}".format(epoch, 100 * val_acc))
        self.save_checkpoint()
