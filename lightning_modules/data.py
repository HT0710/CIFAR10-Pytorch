from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import lightning.pytorch as pl
import os


class CIFAR10DataModule(pl.LightningDataModule):
    """This class defines a PyTorch Lightning data module for the CIFAR10 dataset."""
    
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __init__(self, data_dir: str='datasets', batch_size: int = 32, num_workers: int=os.cpu_count()):
        super(CIFAR10DataModule, self).__init__()
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

    # Download the dataset
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    # Assign train/val datasets for use in dataloaders
    def setup(self, stage: str):
        self.train_data = CIFAR10(self.data_dir, train=True, transform=self.train_transform)
        self.val_data = CIFAR10(self.data_dir, train=False, transform=self.val_transform)
        self.classes = self.train_data.classes

    # Create train dataloader
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, **self.dl_dict, shuffle=True)

    # Create valuation dataloader
    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, **self.dl_dict, shuffle=False)
