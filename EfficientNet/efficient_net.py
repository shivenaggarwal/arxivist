import torch
import torch.nn as nn
from math import ceil

base_model = [
  # expand_ratio, channels, repeats, stride, kernel_size
  [1, 16, 1, 1, 3],
  [6, 24, 2, 2, 3],
  [6, 40, 2, 2, 5],
  [6, 80, 3, 2, 3],
  [6, 112, 3, 1, 5],
  [6, 192, 4, 2, 5],
  [6, 320, 1, 1, 3],
]

phi_values = {
    # (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma; depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    super(ConvBlock, self).__init__()

    # groups controls the connections between inputs and outputs. in_channels and out_channels
    # for eg in depth wise convolution, lets say you have a 3x3 kernel => it will be cube of 3x3x3
    # which would span across height x width x channels,
    # but when we want to do depth wise then we would want to do kernel size of 3 for each filter/channel independently
    # if we set to group=1 then it is a normal ConvBlock
    # if we set it to groups=in_channels then it is a depthwise ConvBlock
    self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
    self.bn = nn.BatchNorm2d(out_channels)
    self.silu = nn.SiLU() # SiLU is the same as Swish, to my understanding it is basically a sigmoid done over on a relu

  def forward(self, x):
    return self.silu(self.bn(self.cnn(x)))

# this block is used to calculate the attention scores of each of the channels
# to my understanding we can think of it in the following intiuition
# say we have an image of a plane and idk a cat so then for each of those image we will get a different attention score because they both have different feature maps
# so maybe for a plane image we have certain parts of the feature have more attention score as comapred to the one on cat image
class SqueezeExcitation(nn.Module):
  def __init__(self, in_channels, reduced_dim):
    super(SqueezeExcitation, self).__init__()

    self.se = nn.Sequential(
      nn.AdaptiveAvgPool2d(1), # eg C x H x W -> C x 1 x 1
      nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
      nn.SiLU(),
      nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return x * self.se(x) # so for each channel it will be multiplied by the value that comes out of the se block => how much that channel is prioritized

class InvertedResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8): # reduction here is for se block and survival_prob is for stochastic depth which is basically like dropout
    super(InvertedResidualBlock, self).__init__()

    self.survival_prob = survival_prob
    self.use_residual = in_channels == out_channels and stride == 1
    hidden_dim = in_channels * expand_ratio
    self.expand = in_channels != hidden_dim
    reduced_dim = int(in_channels / reduction)

    if self.expand:
      self.expand_conv = ConvBlock(
        in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
      )

    self.conv = nn.Sequential(
        ConvBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
        SqueezeExcitation(hidden_dim, reduced_dim),
        nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
      )

  def stochastic_depth(self, x):
    if not self.training:
      return x

    binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob # channels x height x width. returns eitehr 0 or 1 for each of our examples
    return torch.div(x, self.survival_prob) * binary_tensor # so we are dividing by survuval_prob to somehow maintain the magnitude of mean and std during test time which might have changed by adding this dropout

  def forward(self, inputs):
    x = self.expand_conv(inputs) if self.expand else inputs

    if self.use_residual:
      return self.stochastic_depth(self.conv(x)) + inputs
    else:
      return self.conv(x)


class EfficientNet(nn.Module):
  def __init__(self, version, num_classes):
    super(EfficientNet, self).__init__()

    width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
    last_channels = ceil(1280 * width_factor)
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.features = self.create_features(width_factor, depth_factor, last_channels)
    self.classifier = nn.Sequential(
      nn.Dropout(dropout_rate),
      nn.Linear(last_channels, num_classes)
    )

  def calculate_factors(self, version, alpha=1.2, beta=1.1):
    phi, res, drop_rate = phi_values[version]
    depth_factor = alpha ** phi
    width_factor = beta ** phi
    return width_factor, depth_factor, drop_rate

  def create_features(self, width_factor, depth_factor, last_channels):
    channels = int(32 * width_factor)
    features = [ConvBlock(3, channels, kernel_size=3, stride=2, padding=1)]
    in_channels = channels

    for expand_ratio, channels, repeats, stride, kernel_size in base_model:
      out_channels = 4*ceil(int(channels * width_factor) / 4) # because we divide in_channels later on for reduced_dim by 4 so making sure it is divisible by 4 (for increasing the depth)
      layer_repeats = ceil(repeats * depth_factor) # increases the number of layers

      for layer in range(layer_repeats):
        features.append(
          InvertedResidualBlock(in_channels, out_channels, expand_ratio=expand_ratio, stride=stride if layer==0 else 1, kernel_size=kernel_size, padding=kernel_size//2)  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
        )
        in_channels = out_channels

    features.append(ConvBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))

    return nn.Sequential(*features)

  def forward(self, x):
    x = self.pool(self.features(x))
    return self.classifier(x.view(x.shape[0], -1)) # flatten it before sending it to linear layer

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10 # 4 examples in a batch with 10 classes
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(
        version=version,
        num_classes=num_classes,
    ).to(device)

    print(model(x).shape)  # (num_examples, num_classes)

if __name__ == "__main__":
    test()
