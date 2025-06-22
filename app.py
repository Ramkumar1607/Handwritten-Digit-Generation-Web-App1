import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 🧠 Generator architecture (must match training script)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embed(labels)
        x = torch.cat([noise, c], dim=1)
        img = self.model(x)
        return img

# ⚙️ Load trained generator model
@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("cgan_generator.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# 🎨 Show images as MNIST-style grid
def show_images(images):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        img = images[i].detach().numpy().reshape(28, 28)
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)

# 🚀 Streamlit UI
st.title("🧠 Handwritten Digit Generator")
st.write("Generate handwritten digits (0–9) using a trained GAN model.")

digit = st.selectbox("Select a digit (0–9):", list(range(10)))
generate = st.button("Generate 5 Images")

if generate:
    st.write(f"Generating 5 samples of digit `{digit}`...")

    generator = load_generator()
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)

    with torch.no_grad():
        output = generator(z, labels)
        output = output.view(-1, 1, 28, 28)

    show_images(output)
