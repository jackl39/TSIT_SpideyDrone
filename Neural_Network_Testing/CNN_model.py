from torch import nn
import torch

class SpideyDroneNet(nn.Module):
    def __init__(self, inputShape: int, hiddenUnits: int, outputShape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=inputShape,
                    out_channels=hiddenUnits,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.Conv2d(in_channels=hiddenUnits,
                    out_channels=hiddenUnits,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnits,
                    out_channels=hiddenUnits,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.Conv2d(in_channels=hiddenUnits,
                    out_channels=hiddenUnits,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(hiddenUnits, hiddenUnits, 3, padding=1),
            nn.Conv2d(hiddenUnits, hiddenUnits, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hiddenUnits, hiddenUnits, 3, padding=1),
            nn.Conv2d(hiddenUnits, hiddenUnits, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenUnits * 64 * 64,
                    out_features=outputShape)
        )                        
    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        # Uncomment this when running for the first time to determine linear layer's size
        # print(x.shape) 
        x = self.classifier(x)
        return x