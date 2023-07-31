from .base_trainer import BaseAlgorithm
from .utils import train_dcm_single_epoch


class DCM(BaseAlgorithm):
    def __init__(
        self,
        config,
        model,
        calibrator,
        finetune_set_loader,
        uncertainty_set_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        setting,
        checkpoint_path,
        confidence_weight,
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
        self.finetune_set_loader = finetune_set_loader
        self.uncertainty_set_loader = uncertainty_set_loader
        self.confidence_weight = confidence_weight
        print("DCM")

    def run_training(self, num_epochs):
        for epoch in range(num_epochs):
            val_acc = train_dcm_single_epoch(
                model=self.model,
                finetune_set_loader=self.finetune_set_loader,
                uncertainty_set_loader=self.uncertainty_set_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                confidence_weight=self.confidence_weight,
            )
            print("Epoch {}, Validation Accuracy: {:.2f}".format(epoch, 100 * val_acc))
        self.save_checkpoint()
