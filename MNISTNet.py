from typing import List

import torch

class MNISTNet(torch.nn.Module):

    def conv_act(self, in_channels, out_channels, padding):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
            torch.nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1),
            torch.nn.LeakyReLU(negative_slope=0.001)
        )

    def full_convolution_block(self, ch_gradation, padding=(0, 0)):
        return torch.nn.Sequential(
            self.conv_act(ch_gradation[0], ch_gradation[1], padding=padding[0]),
            self.conv_act(ch_gradation[1], ch_gradation[2], padding=padding[1]),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def __init__(self):
        super(MNISTNet, self).__init__()

        channel_gradation = [1, 10, 20, 24, 16]

        self.full_convolution_block1 = self.full_convolution_block(channel_gradation[0:])
        self.full_convolution_block2 = self.full_convolution_block(channel_gradation[2:], padding=(1, 0))

        self.full_connected_block = torch.nn.Sequential(
            torch.nn.Linear(5 * 5 * 16, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.full_convolution_block1(x)
        x = self.full_convolution_block2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.full_connected_block(x)
        return x