from PIL import Image
import matplotlib.pyplot as plt
import re
import os

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

def plot_grid_losses(data_dir, figsize=(12, 9)):
    files = [f for f in os.listdir(data_dir)]
    fig, axs = plt.subplots(-(len(files)//-2), 2, figsize=figsize) # Roligt "ceiling divide" trick!
    axs = axs.flatten()
    
    for i, file in enumerate(files):
        epochs, train_losses, val_losses = extract_data(os.path.join(data_dir, file))
        axs[i].plot(epochs, train_losses, label="Train Loss")
        axs[i].plot(epochs, val_losses, label="Val Loss")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Loss")
        axs[i].set_title(file)
        axs[i].legend()
    
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle("Losses (fig. 2)")
    plt.tight_layout()
    plt.show()
    

def display_images(image_paths, figsize=(10, 10), titles: list = None):
    fig, axs = plt.subplots(-(len(image_paths)//-2), 2, figsize=figsize)
    axs = axs.flatten()
    
    for i, image_path in enumerate(image_paths):
        image = Image.open('imgs/inference/'+image_path)
        axs[i].imshow(image)
        axs[i].axis("off")
        if titles:
            axs[i].set_title(titles[i])
    
    fig.suptitle("Inference Images - Low loss = better (fig. 3)")
    plt.tight_layout()
    plt.show()