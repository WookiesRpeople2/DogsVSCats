import torch
import torch.nn as nn


class NeuralNet(nn.Module):

    def __conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

    def __lin_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = self.__conv_block(in_channels, 16)
        self.conv2 = self.__conv_block(16, 32)
        self.conv3 = self.__conv_block(32, 64)

        self.flatten = nn.Flatten()

        self.ln1 = self.__lin_block(61504, 500)  # after flattening
        self.ln2 = self.__lin_block(500, 50)
        self.ln3 = nn.Linear(50, out_channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.softmax(x)

        return x
