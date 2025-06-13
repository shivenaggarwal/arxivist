import torch
import torch.nn as nn
from torchview import draw_graph

class VGG16(nn.Module):
  def __init__(self, num_classes:int=1000, dropout:float=0.5):
    super(VGG16,self).__init__()
    self.dropout = dropout
    self.num_classes = num_classes

    # feature block
    self.features = nn.Sequential(
      # input shape = (3, 224, 224)
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2), # (64, 112, 112)

      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2), # (128, 56, 56)

      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2), # (256, 28, 28)

      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2), # (512, 14, 14)

      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2), # (512, 7, 7)
    )

    self.classifier = nn.Sequential(
      nn.Linear(in_features=512*7*7, out_features=4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=self.dropout),

      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=dropout),

      nn.Linear(in_features=4096, out_features=self.num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

vgg_model = VGG16()
graph = draw_graph(vgg_model, input_size=(1, 3, 224, 224), expand_nested=True)
graph.visual_graph.render("vgg16_graph", format="png")
