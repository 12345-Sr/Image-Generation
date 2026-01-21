import streamlit as st
import torch
import torch.nn as nn

st.title("🖼️ GAN Image Generator")

device = "cpu"

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

G = Generator().to(device)
G.eval()

if st.button("Generate Image"):
    noise = torch.randn(1,100).to(device)
    img = G(noise).view(28,28).detach().numpy()
    st.image(img, caption="Generated Image", clamp=True)
