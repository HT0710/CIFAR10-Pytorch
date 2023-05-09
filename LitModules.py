import os
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchmetrics import Accuracy
import lightning.pytorch as pl


class CIFARDataModule(pl.LightningDataModule):
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __init__(self, data_dir: str=os.getcwd(), batch_size: int = 32, num_workers: int=os.cpu_count()):
        super().__init__()
        self.data_dir = data_dir
        self.dl_dict = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        # Data transforms
        self.train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.GaussianBlur(kernel_size=3),
            T.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1),
            T.ToTensor(),
            self.normalize
        ])
        self.val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            self.normalize
        ])

    def prepare_data(self):
        # Download the dataset
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train_data = CIFAR10(self.data_dir, train=True, transform=self.train_transform)
        self.val_data = CIFAR10(self.data_dir, train=False, transform=self.val_transform)
        self.classes = self.train_data.classes

    def train_dataloader(self):
        # Create train dataloader
        return DataLoader(dataset=self.train_data, **self.dl_dict, shuffle=True)

    def val_dataloader(self):
        # Create valuation dataloader
        return DataLoader(dataset=self.val_data, **self.dl_dict, shuffle=False)


class LitModel(pl.LightningModule):
    def __init__(self, model, lr: float=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.accuracy = Accuracy('multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3),
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log_dict({"train_loss": loss, "train_acc": acc})
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log_dict({"val_loss": loss, "val_acc": acc})
        return loss
