import matplotlib.pyplot as plt
import re
import torch
import os

from utils import gpu_transform

def plot_loss(epochs, train_losses, val_losses, title=None):
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()

def extract_data(file):
    with open(file, "r") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]

    epochs = []
    train_losses = []
    val_losses = []

    for line in lines:
        if line.strip():
            m = re.search(r'Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
    
    return epochs, train_losses, val_losses

def plot_keypoints(image):
    test_pic = image

    test_image = plt.imread(test_pic)
    test_image = gpu_transform(test_image, "cuda").unsqueeze(0).to(device)

    with torch.no_grad():
        pred_keypoints = model(test_image)
        
    pred_keypoints_np = pred_keypoints.cpu().numpy().reshape(-1, 2)
    pred_keypoints_np[:, 0] *= test_image.shape[3]
    pred_keypoints_np[:, 1] *= test_image.shape[2]

    fig, ax = plt.subplots(1)
    ax.imshow(test_image.cpu().squeeze().permute(1, 2, 0))

    for (x, y) in pred_keypoints_np:
        ax.plot(x, y, 'go')

    plt.show()