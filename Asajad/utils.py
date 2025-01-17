import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2


ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(filename='data/Annotations/skipped_images.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FaceKeypointDataset(Dataset):
    """
    Custom Dataset for Facial Keypoint Detection using Albumentations for data augmentation.
    
    Args:
        annotations (pd.DataFrame): DataFrame containing image file names and keypoints.
        root_dir (str): Directory with all the images.
        transform (albumentations.Compose, optional): Albumentations transformations to apply.
    """
    def __init__(self, annotations, root_dir, transform=None):
        self.annotations = annotations.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.annotations.iloc[idx]['file_id']
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        keypoints = self.annotations.iloc[idx][['LeftEye_x', 'LeftEye_y', 
                     'RightEye_x', 'RightEye_y',
                     'Nose_x', 'Nose_y',  
                     'Mouth_x', 'Mouth_y']].values.astype('float')

        # Normalize keypoints to [0, 1]
        w, h = image.size
        keypoints = keypoints / np.array([w, h] * 4)

        keypoints = keypoints.reshape(-1, 2).tolist()

        if self.transform:
            transformed = self.transform(image=np.array(image), keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints']

        else:
            image = ToTensorV2()(image=np.array(image))['image']

        keypoints = torch.tensor(keypoints).float().flatten()  # Shape: (8,)

        return image, keypoints

def get_train_transforms():
    """
    Defines the data augmentation pipeline using Albumentations.
    
    Returns:
        albumentations.Compose: Composed transformations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_validation_transforms():
    """
    Defines the validation/test data preprocessing pipeline using Albumentations.
    
    Returns:
        albumentations.Compose: Composed transformations.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def visualize_inference(model, dataset, device='cuda', num_samples=5):
    """
    Performs inference on a specified number of test samples and visualizes the predicted keypoints.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (torch.utils.data.Dataset): The test dataset.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        num_samples (int): Number of test samples to visualize.
    """
    model.to(device)
    model.eval()
    
    for i in range(num_samples):
        sample = dataset[i]
        if sample is None:
            print(f"Sample {i+1} is None. Skipping.")
            continue
        image, _ = sample
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_keypoints = model(image_tensor)
        
        image_shape = image.shape[1:]
        pred_keypoints = pred_keypoints.cpu().numpy().reshape(-1, 2)
        pred_keypoints[:, 0] = pred_keypoints[:, 0] * image_shape[1]
        pred_keypoints[:, 1] = pred_keypoints[:, 1] * image_shape[0]
        
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy().transpose((1, 2, 0))
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.array(image)
        
        plt.figure(figsize=(6,6))
        plt.imshow(image_np)
        
        plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c='r', s=20, marker='x', label='Predicted Keypoints')
        
        plt.title(f"Test Sample {i+1}")
        plt.axis('off')
        plt.legend()
        plt.show()



def visualize_augmented_samples(dataset, num_samples=5):
    """
    Visualizes a specified number of augmented samples from the dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to visualize samples from.
        num_samples (int): Number of samples to visualize.
    """
    for i in range(num_samples):
        image, keypoints = dataset[i]
        image_np = image.permute(1, 2, 0).numpy()
        # Denormalize for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        keypoints_np = keypoints.numpy().reshape(-1, 2)  # Shape: (4,2)
        # Convert back to original scale
        h, w, _ = image_np.shape
        keypoints_np[:, 0] *= w
        keypoints_np[:, 1] *= h
        
        plt.figure(figsize=(6,6))
        plt.imshow(image_np)
        plt.scatter(keypoints_np[:, 0], keypoints_np[:, 1], c='r', s=20, marker='x')
        plt.title(f"Augmented Sample {i+1}")
        plt.axis('off')
        plt.show()

def train(model, criterion_keypoints, optimizer, scheduler, train_loader, val_loader, epochs=50, device='cuda', patience=10):
    """
    Trains the model on the provided dataset with progress bars, early stopping, and checkpointing.
    
    Args:
        model (torch.nn.Module): The neural network model.
        criterion_keypoints (torch.nn.Module): Loss function for keypoints.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        device (str): Device to train on ('cuda' or 'cpu').
        patience (int): Number of epochs with no improvement after which training will be stopped.
    
    Returns:
        torch.nn.Module: The trained model with the best validation loss.
    """
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None
    history_data = []

    for epoch in range(epochs):
        epoch_num = epoch + 1
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        # Training Phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{epochs} - Training", leave=False)
        for batch in train_bar:
            if batch[0] is None:
                continue
            images, landmarks = batch
            images, landmarks = images.to(device), landmarks.to(device)

            optimizer.zero_grad()
            keypoints_pred = model(images)

            loss = criterion_keypoints(keypoints_pred, landmarks.view(-1, 8))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix({'Batch Loss': loss.item()})

        # Validation Phase
        model.eval()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch_num}/{epochs} - Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                if batch[0] is None:
                    continue
                images, landmarks = batch
                images, landmarks = images.to(device), landmarks.to(device)

                keypoints_pred = model(images)

                loss = criterion_keypoints(keypoints_pred, landmarks.view(-1, 8))

                val_loss += loss.item()

                val_bar.set_postfix({'Val Loss': loss.item()})

        if scheduler is not None:
            scheduler.step(val_loss / len(val_loader))

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch_num}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        history_data.append({
            'epoch': epoch_num,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, 'best_keypoint_model.pth')
            print("Validation loss decreased. Saving model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    history_df = pd.DataFrame(history_data)
    return model, history_df
