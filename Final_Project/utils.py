import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_log_dir(root_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(root_dir, f"logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def plot_metrics(log_dir, loss_list, acc_list, epochs, mode="Train"):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{mode} Loss Over Epochs")
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_list, marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{mode} Accuracy Over Epochs")
    plt.grid(True)

    plt.tight_layout()

    save_path = os.path.join(log_dir, f"{mode.lower()}_metrics.png")
    plt.savefig(save_path)
    plt.close()
    print(f"{mode} metrics plot saved to {save_path}")

def compute_mean_std(dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    squared_mean = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        # images: (B, C, H, W)
        images = images.float() / 255.0 if images.max() > 1 else images
        batch, C, H, W = images.shape
        
        images = images.view(batch, C, -1)  # (B, C, N)
        pixels = batch * H * W

        mean += images.mean(dim=[0,2]) * pixels
        squared_mean += (images ** 2).mean(dim=[0,2]) * pixels
        total_pixels += pixels

    mean /= total_pixels
    squared_mean /= total_pixels
    std = torch.sqrt(squared_mean - mean ** 2)

    return mean, std
