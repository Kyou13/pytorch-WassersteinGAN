import torch
from torch import nn


class Discriminator(nn.Module):
  def __init__(self, imageSize):
    super(Discriminator, self).__init__()
    self.linear1 = nn.Linear(imageSize, 512)
    self.linear2 = nn.Linear(512, 256)
    self.linear3 = nn.Linear(256, 1)
    self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    x = self.leakyReLU(self.linear1(x))
    x = self.leakyReLU(self.linear2(x))
    x = self.linear3(x)
    return torch.squeeze(x)


class Generator(nn.Module):
  def __init__(self, nz, imageSize):
    super(Generator, self).__init__()
    self.linear1 = nn.Linear(nz, 128)
    self.linear2 = nn.Linear(128, 256)
    self.bn1 = nn.BatchNorm1d(256, 0.8)
    self.linear3 = nn.Linear(256, 512)
    self.bn2 = nn.BatchNorm1d(512, 0.8)
    self.linear4 = nn.Linear(512, 1024)
    self.bn3 = nn.BatchNorm1d(1024, 0.8)
    self.linear5 = nn.Linear(1024, imageSize)
    self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    x = self.leakyReLU(self.linear1(x))
    x = self.leakyReLU(self.bn1(self.linear2(x)))
    x = self.leakyReLU(self.bn2(self.linear3(x)))
    x = self.leakyReLU(self.bn3(self.linear4(x)))
    x = torch.tanh(self.linear5(x))
    return x
