import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from parso.normalizer import Normalizer
from dill.temp import b

class Discriminator(nn.Module):
  def __init__(self, in_features):
    super().__init__()
    self.disc = nn.Sequential(
      nn.Linear(in_features, 128),
      nn.LeakyReLU(0.1),
      nn.Linear(128, 1), # doing this because this will result in a single value output fake or real
      nn.Sigmoid(), # obviously so that it is between 0 and 1
    )

  def forward(self, x):
    return self.disc(x)

class Generator(nn.Module):
  def __init__(self, z_dim, img_dim):
    super().__init__()
    self.gen = nn.Sequential(
      nn.Linear(z_dim, 256),
      nn.LeakyReLU(0.1),
      nn.Linear(256, img_dim), # from MNIST 28 * 28 * 1 -> 784
      nn.Tanh(),
    )
  def forward(self, x):
    return self.gen(x)

# Set device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set hyperparameters
lr = 3e-4 # works best with adam imo
z_dim = 64 # 128, 256
img_dim = 784 # 28 * 28 * 1
batch = 32
epochs = 50

# Initialize Discriminator and Generator
disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)

# Create fixed noise for visualization
fixed_noise = torch.randn((batch, z_dim)).to(device)

# Define data transformations
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load dataset
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch, shuffle=True)

# Load Optimizer
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# Define loss function
criterion = nn.BCELoss()

# Initialize TensorBoard writers
writer_r = SummaryWriter(f"runs/GAN/real")
writer_f = SummaryWriter(f"runs/GAN/fake")
step = 0

# Training loop
for epoch in range(epochs):
  for batch_idx, (real, _) in enumerate(loader):
    real = real.view(-1, 784).to(device) # -1 means we keep the number of examples in our batch and then we flatten the rest to 784
    batch = real.shape[0]

    # Train Discriminator: max(log(D(real))) + log(1-D(G(z)))
    noise = torch.randn(batch, z_dim).to(device) # from gaussian dist of mean 0 and std bw 0 and 1
    fake = gen(noise)

    disc_real = disc(real).view(-1)
    lossD_real = criterion(disc_real, torch.ones_like(disc_real))
    disc_fake = disc(fake).view(-1)
    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    lossD = (lossD_fake + lossD_real) / 2
    disc.zero_grad()
    lossD.backward(retain_graph=True) # this wont clear the intermediate computation which will help in retaining values which we use for generator
    opt_disc.step()

    # Train Generator: min(log(1-D(G(z)))) ==> max(log(D(G(z))))
    out = disc(fake).view(-1) # we need intermidate results from this fake which are retained above in the computational graph
    lossG = criterion(out, torch.ones_like(out))
    gen.zero_grad()
    lossG.backward()
    opt_gen.step()

    # Print progress and generate images for TensorBoard
    if batch_idx == 0:
      print(f"Epoch : {epoch}/{epochs} | Loss Disc : {lossD:.4f} | Loss Gen : {lossG:.4f}")
      with torch.no_grad():
        fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
        data = real.reshape(-1, 1, 28, 28)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(data, normalize=True)
        writer_f.add_image("Fake Images", img_grid_fake, global_step=step)
        writer_r.add_image("Real Images", img_grid_real, global_step=step)
        step += 1
