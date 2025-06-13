import torch
import torch.nn as nn


# input_img -> hidden_dim -> mean, std -> parametrization trick -> mean + std dot epsilon -> decoder -> output_img
class VAE(nn.Module):
  def __init__(self, input_dim, h_dim=200, z_dim=20):
    super().__init__()

    # for the encoder
    self.img2hid = nn.Linear(input_dim, h_dim)
    self.hid2mu = nn.Linear(h_dim, z_dim)
    self.hid2sigma = nn.Linear(h_dim, z_dim)

    # for the decoder
    self.z2hid = nn.Linear(z_dim, h_dim)
    self.hid2img = nn.Linear(h_dim, input_dim)

    self.relu = nn.ReLU()

  def encode(self, x):
    # q_phi(z|x)
    h = self.relu(self.img2hid(x))
    mu, sigma = self.hid2mu(h), self.hid2sigma(h)
    return mu, sigma

  def decode(self, z):
    # p_theta(x|z)
    h = self.relu(self.z2hid(z))
    return torch.sigmoid(self.hid2img(h))

  def forward(self, x):
    mu, sigma = self.encode(x)
    epsilon = torch.randn_like(sigma)
    z_reparametrized = mu + sigma*epsilon  # (element wise multiplication)
    x_reconstructed = self.decode(z_reparametrized)
    return x_reconstructed, mu, sigma # mu and sigma for kl divergence and x_reconstructed for the loss

if __name__ == "__main__":
  x = torch.randn(32, 28*28) # 28 * 28 = 784 for mnist dataset
  vae = VAE(input_dim=784)
  x_reconstructed, mu, sigma = vae(x)
  print(x_reconstructed.shape)
  print(mu.shape)
  print(sigma.shape)
