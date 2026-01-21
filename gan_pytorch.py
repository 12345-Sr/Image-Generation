import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

os.makedirs("generated_images", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# DATA
# =============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(
    root=".", download=True, transform=transform
)

loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# =============================
# MODELS
# =============================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
g_opt = torch.optim.Adam(G.parameters(), lr=1e-4)
d_opt = torch.optim.Adam(D.parameters(), lr=1e-4)

# =============================
# TRAINING
# =============================
for epoch in range(20):
    print(f"\nEpoch {epoch+1}/20")
    for imgs, _ in tqdm(loader):
        imgs = imgs.view(-1, 784).to(device)
        bs = imgs.size(0)

        real = torch.ones(bs,1).to(device)
        fake = torch.zeros(bs,1).to(device)

        # Train Discriminator
        noise = torch.randn(bs,100).to(device)
        gen_imgs = G(noise)

        d_loss = loss_fn(D(imgs), real) + loss_fn(D(gen_imgs.detach()), fake)
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # Train Generator
        g_loss = loss_fn(D(gen_imgs), real)
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    with torch.no_grad():
        img = gen_imgs[0].view(28,28).cpu()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.savefig(f"generated_images/epoch_{epoch+1}.png")
        plt.close()
