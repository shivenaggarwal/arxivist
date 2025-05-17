import torch
import torch.nn as nn
from torchview import draw_graph


class Unet(nn.Module):
  def __init__(self, in_channels=1,  out_channels=2):
    super(Unet, self).__init__()

    # Encoder
    self.enc1 = self.conv_block(in_channels, 64)
    self.enc2 = self.conv_block(64, 128) # all the values are from the paper
    self.enc3 = self.conv_block(128, 256)
    self.enc4 = self.conv_block(256, 512)

    self.bottleneck = self.conv_block(512, 1024)

    # Decoder
    self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # upsampling
    self.dec4 = self.conv_block(1024, 512)
    self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.dec3 = self.conv_block(512, 256)
    self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec2 = self.conv_block(256, 128)
    self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec1 = self.conv_block(128, 64)

    self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

  def conv_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3),
      nn.ReLU(inplace=True),
    )

  def crop(self, enc_op, dec_op): # first block of enc and dec and we are taking the output of it
    enc_op_size = enc_op.size()[2:] # returns height, width since our output is of shape (batch_size, in_channels, height, width)
    dec_op_size = dec_op.size()[2:]

    delta_h = enc_op_size[0] - dec_op_size[0] # subtracting the size of encoder op height by dec op height
    delta_w = enc_op_size[1] - dec_op_size[1] # subtracting the size of encoder op width by dec op width

    return enc_op[:, :, delta_h//2:enc_op_size[0]-delta_h//2, delta_w//2:enc_op_size[1]-delta_w//2]

  def forward(self, x):

    # Encoder block
    enc1 = self.enc1(x)
    enc2 = self.enc2(nn.MaxPool2d(kernel_size=2)(enc1))
    enc3 = self.enc3(nn.MaxPool2d(kernel_size=2)(enc2))
    enc4 = self.enc4(nn.MaxPool2d(kernel_size=2)(enc3))
    bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc4))

    # Decoder block
    dec4 = self.upconv4(bottleneck)
    dec4 = self.dec4(torch.cat((dec4, self.crop(enc4, dec4)), dim=1))
    dec3 = self.upconv3(dec4)
    dec3 = self.dec3(torch.cat((dec3, self.crop(enc3, dec3)), dim=1))
    dec2 = self.upconv2(dec3)
    dec2 = self.dec2(torch.cat((dec2, self.crop(enc2, dec2)), dim=1))
    dec1 = self.upconv1(dec2)
    dec1 = self.dec1(torch.cat((dec1, self.crop(enc1, dec1)), dim=1))

    return self.final_layer(dec1)

if __name__ == "__main__":
    model = Unet()
    graph = draw_graph(model, input_size=(1, 1, 572, 572), expand_nested=True)
    graph.visual_graph.view()
