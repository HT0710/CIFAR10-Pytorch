from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchmetrics import Accuracy
import torch.nn.functional as F
import lightning.pytorch as pl


class LitModel(pl.LightningModule):
    """PyTorch Lightning module"""
    
    def __init__(self, model, lr: float=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.accuracy = Accuracy('multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    # Log learning rate (remove soon -> move to callback.py)
    def on_train_epoch_start(self):
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr)

    # Setup the optimizer
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.1, patience=3),
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    # Train step
    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log_dict({"train_loss": loss, "train_acc": acc})
        return loss

    # Validate step
    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log_dict({"val_loss": loss, "val_acc": acc})
        return loss
