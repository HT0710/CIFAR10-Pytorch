import torch.nn as nn
from torchinfo import summary


config = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes: int=10):
        super(VGG, self).__init__()
        self.features = self._construction(vgg_name)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def _construction(self, name):
        sequence = nn.Sequential()
        in_channels = 3
        for x in config[name]:
            if x == 'M':
                sequence.extend([
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ])
            else:
                sequence.extend([
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(True)
                ])
                in_channels = x
        return sequence

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


if __name__=='__main__':
    summary(VGG('VGG16'), [1, 3, 224, 224])
