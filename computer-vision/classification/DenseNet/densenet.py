import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph

class Bottleneck(nn.Module):
  def __init__(self, in_channels, growth_rate): # growth rate is basically output in 32 channels
    super(Bottleneck, self).__init__()

    # 1x1
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1)

    # 3x3
    self.bn2 = nn.BatchNorm2d(4 * growth_rate)
    self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)

  def forward(self, x):

    out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))

    out = torch.cat([x, out], 1) # dim 1 concat at feature channel

    return out

class DenseBlock(nn.Module):
  def __init__(self, num_layers, in_channels, growth_rate):
    super(DenseBlock, self).__init__()

    layers = []

    for i in range(num_layers):
      layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))

    self.dense_block = nn.Sequential(*layers)

  def forward(self, x):
    return self.dense_block(x)

class Transition(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Transition, self).__init__()

    self.bn = nn.BatchNorm2d(in_channels) # eg. 56 X 56 X 380 (not values from paper just for understanding)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # eg. 56 X 56 X 196
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # => 28 X 28 X 196

  def forward(self, x):
    x = self.conv(F.relu(self.bn(x)))
    x = self.pool(x)
    return x

class DenseNet(nn.Module):
  def __init__(self, in_channels, num_blocks:list, growth_rate:int=32, num_classes=1000):
    super(DenseNet, self).__init__()

    self.growth_rate = growth_rate
    self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm2d(2 * growth_rate)
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    num_channels = 2 * growth_rate

    # Dense Block 1
    self.dense1 = DenseBlock(num_blocks[0], num_channels, growth_rate)
    num_channels += num_blocks[0] * growth_rate
    # Transition layer 1
    self.trans1 = Transition(num_channels, num_channels // 2)
    num_channels = num_channels // 2

    # Dense Block 2
    self.dense2 = DenseBlock(num_blocks[1], num_channels, growth_rate)
    num_channels += num_blocks[1] * growth_rate
    # Transition layer 2
    self.trans2 = Transition(num_channels, num_channels // 2)
    num_channels = num_channels // 2

    # Dense Block 3
    self.dense3 = DenseBlock(num_blocks[2], num_channels, growth_rate)
    num_channels += num_blocks[2] * growth_rate
    # Transition layer 3
    self.trans3 = Transition(num_channels, num_channels // 2)
    num_channels = num_channels // 2

    # Dense Block 4
    self.dense4 = DenseBlock(num_blocks[3], num_channels, growth_rate)
    num_channels += num_blocks[3] * growth_rate

    self.bn2 = nn.BatchNorm2d(num_channels)
    self.fc = nn.Linear(num_channels, num_classes)

  def forward(self, x):
    x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    x = self.dense1(x)
    x = self.trans1(x)
    x = self.dense2(x)
    x = self.trans2(x)
    x = self.dense3(x)
    x = self.trans3(x)
    x = self.dense4(x)
    x = F.relu(self.bn2(x))
    x = F.adaptive_avg_pool2d(x, (1,1)) # 128 x 7 x 7 => 128 x 1 x 1
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x


model = DenseNet(in_channels=3, num_blocks=[6, 12, 24, 16], growth_rate=32)
viz = draw_graph(model, input_size=(1, 3, 224, 224), expand_nested=True)
viz.visual_graph.view()  # Opens the image in the default viewer
