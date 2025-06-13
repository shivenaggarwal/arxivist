import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the basic convolution block
class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super(ConvBlock, self).__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.bn(self.conv(x)))

# Building the inception block
class Inception(nn.Module):
  def __init__(self, in_channels, num1x1, num3x3_reduce, num3x3, num5x5_reduce, num5x5, pool_proj):
    super(Inception, self).__init__()

    # 4 output channel for each parallel block of network
    # blocks are ran parallely not sequentially
    self.block1 = nn.Sequential(
      ConvBlock(in_channels, num1x1, kernel_size=1, stride=1, padding=0)
    )
    self.block2 = nn.Sequential(
      ConvBlock(in_channels, num3x3_reduce, kernel_size=1, stride=1, padding=0),
      ConvBlock(num3x3_reduce, num3x3, kernel_size=3, stride=1, padding=1)
    )
    self.block3 = nn.Sequential(
      ConvBlock(in_channels, num5x5_reduce, kernel_size=1, stride=1, padding=0),
      ConvBlock(num5x5_reduce, num5x5, kernel_size=5, stride=1, padding=2)
    )
    self.block4 = nn.Sequential(
      nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
      ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
    )

  def forward(self, x):
    block1 = self.block1(x)
    block2 = self.block2(x)
    block3 = self.block3(x)
    block4 = self.block4(x)

    return torch.cat([block1, block2, block3, block4], 1) # N * filters * 28 * 28 here 1 denotes that we are concatinating the filter size

class Auxiliary(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(Auxiliary, self).__init__()

    self.pool = nn.AdaptiveAvgPool2d((4,4))
    self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(2048, 1024)
    self.dropout = nn.Dropout(0.7)
    self.fc2 = nn.Linear(1024, num_classes)

  def forward(self, x):
    out = self.pool(x)
    out = self.conv(out)
    out = self.relu(out)
    # ao, a conv output of [batch_size, num_channel, height, width] should be “flattened” to become a `[batch_size, num_channel x height x width]` tensor.
    # and the in_`features` of the linear layer should be set to `[num_channel * height * width]`
    out = torch.flatten(out,1) # conv will output a 4 dim tensor but fc1 requires 2 dim vector
    out = self.fc1(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.fc2(out)

    return out

class GoogLeNet(nn.Module):
  def __init__(self, num_classes=10):
    super(GoogLeNet, self).__init__()

    self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
    self.pool1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
    self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0)
    self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
    self.pool3 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

    self.inception3A = Inception(in_channels=192, num1x1=64, num3x3_reduce=96, num3x3=128, num5x5_reduce=16, num5x5=32, pool_proj=32)
    self.inception3B = Inception(in_channels=256, num1x1=128, num3x3_reduce=128, num3x3=192, num5x5_reduce=32, num5x5=96, pool_proj=64)
    self.pool4 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

    self.inception4A = Inception(in_channels=480, num1x1=192, num3x3_reduce=96, num3x3=208, num5x5_reduce=16, num5x5=48, pool_proj=64)
    self.inception4B = Inception(in_channels=512, num1x1=160, num3x3_reduce=112, num3x3=224, num5x5_reduce=24, num5x5=64, pool_proj=64)
    self.inception4C = Inception(in_channels=512, num1x1=128, num3x3_reduce=128, num3x3=256, num5x5_reduce=24, num5x5=64, pool_proj=64)
    self.inception4D = Inception(in_channels=512, num1x1=112, num3x3_reduce=144, num3x3=288, num5x5_reduce=32, num5x5=64, pool_proj=64)
    self.inception4E = Inception(in_channels=528, num1x1=256, num3x3_reduce=160, num3x3=320, num5x5_reduce=32, num5x5=128, pool_proj=128)
    self.pool5 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

    self.inception5A = Inception(in_channels=832, num1x1=256, num3x3_reduce=160, num3x3=320, num5x5_reduce=32, num5x5=128, pool_proj=128)
    self.inception5B = Inception(in_channels=832, num1x1=384, num3x3_reduce=192, num3x3=384, num5x5_reduce=48, num5x5=128, pool_proj=128)
    self.pool6 = nn.AdaptiveAvgPool2d((1, 1))

    self.dropout = nn.Dropout(0.4)
    self.fc = nn.Linear(1024, num_classes)
    self.aux4A = Auxiliary(512, num_classes)
    self.aux4D = Auxiliary(528, num_classes)

  def forward(self, x):
    out = self.conv1(x)
    out = self.pool1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.pool3(out)
    out = self.inception3A(out)
    out = self.inception3B(out)
    out = self.pool4(out)
    out = self.inception4A(out)
    aux1 = self.aux4A(out)
    out = self.inception4B(out)
    out = self.inception4C(out)
    out = self.inception4D(out)
    aux2 = self.aux4D(out)
    out = self.inception4E(out)
    out = self.pool5(out)
    out = self.inception5A(out)
    out = self.inception5B(out)
    out = self.pool6(out)
    out = torch.flatten(out, 1)
    out = self.dropout(out)
    out = self.fc(out)

    return out, aux1, aux2
