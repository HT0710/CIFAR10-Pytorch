import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T


def create_dataloaders(batch_size: int = 32, num_worker: int = os.cpu_count()):
    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.GaussianBlur(kernel_size=3),
        T.ColorJitter(brightness=0.4,
                      contrast=0.4,
                      saturation=0.4,
                      hue=0.1),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
    ])

    if not os.path.exists('cifar-10-batches-py'):
        download = True
    else:
        download = False

    train_data = CIFAR10('.', train=True, download=download, transform=train_tf)
    val_data = CIFAR10('.', train=False, download=download, transform=val_tf)

    # Create Dataloader
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=num_worker,
                              shuffle=True)
    
    val_loader = DataLoader(val_data,
                             batch_size=batch_size,
                             num_workers=num_worker,
                             shuffle=False)

    return train_loader, val_loader
