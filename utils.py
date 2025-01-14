# utils.py

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import os
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from PIL import Image, ImageFile
import logging

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging for skipped images
logging.basicConfig(filename='data/Annotations/skipped_images.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

class FaceKeypointDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None):
        """
        Args:
            annotations (pd.DataFrame): DataFrame containing keypoint annotations.
            root_dir (str): Directory with all the preprocessed images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.annotations = annotations  # pandas dataframe
        self.root_dir = root_dir        # path to preprocessed images
        self.transform = transform      # transform operations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.root_dir, self.annotations["file_id"][idx])

            # Open the image using PIL
            with Image.open(img_name) as img:
                img = img.convert('RGB')  # Ensure image has 3 channels

            # Keypoints
            landmarks_cols = ["LeftEye_x","LeftEye_y",
                              "RightEye_x","RightEye_y",
                              "Nose_x","Nose_y",
                              "Mouth_x","Mouth_y"]
            landmarks = self.annotations.loc[idx, landmarks_cols].values
            landmarks = landmarks.astype('float')  # Shape: (8,)

            if self.transform:
                image = self.transform(img)
            else:
                # If no transform is provided, convert PIL image to tensor
                image = transforms.ToTensor()(img)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])(image)

            # Convert landmarks to tensor
            landmarks = torch.tensor(landmarks, dtype=torch.float32)

            return image, landmarks

        except Exception as e:
            logging.info(f"Skipping {idx} due to error: {e}")
            return None

def skip_none_collate_fn(batch):
    """
    Collate function that filters out None samples and collates the rest.
    Returns:
        images: Tensor of shape (batch_size, 3, 224, 224)
        landmarks: Tensor of shape (batch_size, 8)
    """
    # Filter out any items that are None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    images, landmarks = zip(*batch)
    images = default_collate(images)
    landmarks = default_collate(landmarks)
    return images, landmarks

def train_test_split(dataset, train_size=0.8, val_size=0.1, batch_size=32):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_size (float): Proportion of the dataset to include in the training set.
        val_size (float): Proportion of the dataset to include in the validation set.
        batch_size (int): Number of samples per batch.

    Returns:
        train_loader, val_loader, test_loader (DataLoader): DataLoaders for each split.
    """
    total_size = len(dataset)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=skip_none_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=skip_none_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=skip_none_collate_fn)

    return train_loader, val_loader, test_loader

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

    for epoch in range(epochs):
        epoch_num = epoch + 1
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        # Training Phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{epochs} - Training", leave=False)
        for batch in train_bar:
            if batch[0] is None:
                continue  # Skip batches where all items were None
            images, landmarks = batch
            images, landmarks = images.to(device), landmarks.to(device)

            optimizer.zero_grad()
            keypoints_pred = model(images)

            # Compute loss
            loss = criterion_keypoints(keypoints_pred, landmarks.view(-1, 8))  # Ensure shape matches [Batch, 8]

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update tqdm description with current loss
            train_bar.set_postfix({'Batch Loss': loss.item()})

        # Validation Phase
        model.eval()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch_num}/{epochs} - Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                if batch[0] is None:
                    continue  # Skip batches where all items were None
                images, landmarks = batch
                images, landmarks = images.to(device), landmarks.to(device)

                keypoints_pred = model(images)

                # Compute loss
                loss = criterion_keypoints(keypoints_pred, landmarks.view(-1, 8))  # Ensure shape matches [Batch, 8]

                val_loss += loss.item()

                # Update tqdm description with current validation loss
                val_bar.set_postfix({'Val Loss': loss.item()})

        # Scheduler Step (if using a scheduler based on validation loss)
        if scheduler is not None:
            scheduler.step(val_loss / len(val_loader))

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch_num}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, 'best_keypoint_model.pth')  # Save the best model
            print("Validation loss decreased. Saving model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model
