import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Avoid crashing on truncated images


##########################################################
# 1. Define Transformations (Image -> 224x224)
##########################################################

# Using ImageNet mean and standard deviation for normalization

transform = transforms.Compose([
    transforms.Resize(max_size=224, interpolation=Image.LANCZOS, size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

##########################################################
# 2. Dataset Definition with Normalization to [0,1]
##########################################################

class FaceKeypointDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=transform):
        """
        annotations: pandas DataFrame containing file IDs, landmarks, and bboxes
        root_dir:    string path to the folder containing images
        transform:   torchvision transforms to apply to each image
        """
        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.root_dir, self.annotations["file_id"][idx])
            with Image.open(img_name) as img:
                # Convert to RGB to ensure 3 channels
                img = img.convert('RGB')
                # Get original width and height BEFORE resizing
                orig_w, orig_h = img.size

            # Landmarks
            landmarks_cols = [
                "LeftEye_x", "LeftEye_y",
                "RightEye_x", "RightEye_y",
                "Nose_x",     "Nose_y",
                "Mouth_x",    "Mouth_y"
            ]
            landmarks = self.annotations.loc[idx, landmarks_cols].values.astype('float').reshape(-1, 2)

            # Bounding Boxes
            bboxes_cols = ['X_box', 'Y_box', 'W_box', 'H_box']
            bboxes = self.annotations.loc[idx, bboxes_cols].values.astype('float').reshape(-1, 4)

            # --- Apply image transforms (resize to 224x224, etc.) ---
            if self.transform:
                img_transformed = self.transform(img)
            else:
                img_transformed = transforms.ToTensor()(img)

            # --------------------------------------------------------
            # Normalize keypoints and bounding boxes to [0, 1]
            # --------------------------------------------------------
            landmarks[:, 0] /= float(orig_w)  # x / orig_w
            landmarks[:, 1] /= float(orig_h)  # y / orig_h

            bboxes[:, 0] /= float(orig_w)  # X_box / orig_w
            bboxes[:, 1] /= float(orig_h)  # Y_box / orig_h
            bboxes[:, 2] /= float(orig_w)  # W_box / orig_w
            bboxes[:, 3] /= float(orig_h)  # H_box / orig_h

            # Return the image, normalized keypoints, and normalized bounding boxes
            return (
                img_transformed,
                torch.tensor(landmarks, dtype=torch.float32),
                torch.tensor(bboxes, dtype=torch.float32)
            )

        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
            return None

##########################################################
# 3. Custom Collate Function (skip None items)
##########################################################

def skip_none_collate_fn(batch):
    """
    Removes None items from the batch. Returns None if entire batch is invalid.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    images, landmarks, bboxes = zip(*batch)
    images = default_collate(images)
    landmarks = default_collate(landmarks)
    bboxes = default_collate(bboxes)
    return images, landmarks, bboxes

##########################################################
# 4. Split into Train/Val/Test & Move to Device
##########################################################

def train_test_split(dataset, train_size=0.8, val_size=0.1, batch_size=96, device='cuda'):
    """
    Splits the dataset into train, val, test subsets and creates DataLoaders.
    """
    total_size = len(dataset)
    train_count = int(train_size * total_size)
    val_count = int(val_size * total_size)
    test_count = total_size - train_count - val_count

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_count, val_count, test_count])

    # Create DataLoaders
    num_workers = min(os.cpu_count(), 8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=skip_none_collate_fn, pin_memory=True,
        prefetch_factor=4, num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=skip_none_collate_fn, pin_memory=True,
        prefetch_factor=4, num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=skip_none_collate_fn, pin_memory=True,
        prefetch_factor=4, num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


##########################################################
# 5. Training Loop
##########################################################

def train(
    model,
    criterion_keypoints,
    criterion_bbox,
    optimizer,
    train_loader,
    val_loader,
    epochs=50,
    device='cuda',
    bbox_weight=1.0,
    save_dir="./model/models/",
    save_best=True
):
    """
    Trains the model using the specified keypoint and bounding box loss functions.
    Saves model checkpoints each epoch, and optionally saves the best model based on validation loss.
    """
    # Move the model to the GPU/CPU device
    model.to(device)
    
    # Create directory to save models
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_num = epoch + 1
        model.train()
        train_loss = 0.0
        train_steps = 0
        val_loss = 0.0

        # Training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{epochs} - Training", leave=False)
        for batch in train_bar:
            # Move each batch to the specified device
            images, landmarks, bboxes = (
                batch[0].to(device, non_blocking=True),
                batch[1].to(device, non_blocking=True),
                batch[2].to(device, non_blocking=True),
            )

            optimizer.zero_grad()
            keypoints_pred, bboxes_pred = model(images)

            # Compute losses
            loss_keypoints = criterion_keypoints(keypoints_pred, landmarks.view(-1, 8))
            loss_bbox = criterion_bbox(bboxes_pred, bboxes.view(-1, 4))
            loss = loss_keypoints + bbox_weight * loss_bbox

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            train_bar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch_num}/{epochs} - Validation", leave=False)
        val_steps = 0
        with torch.no_grad():
            for batch in val_bar:
                # Move validation batch to the device
                images, landmarks, bboxes = (
                    batch[0].to(device, non_blocking=True),
                    batch[1].to(device, non_blocking=True),
                    batch[2].to(device, non_blocking=True),
                )
                keypoints_pred, bboxes_pred = model(images)

                # Compute validation loss
                loss_keypoints = criterion_keypoints(keypoints_pred, landmarks.view(-1, 8))
                loss_bbox = criterion_bbox(bboxes_pred, bboxes.view(-1, 4))
                loss = loss_keypoints + bbox_weight * loss_bbox

                val_loss += loss.item()
                val_steps += 1
                val_bar.set_postfix({'Val Loss': f"{loss.item():.4f}"})

        # Calculate average losses for the epoch
        avg_train_loss = train_loss / train_steps if train_steps > 0 else float('inf')
        avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')

        print(f"Epoch {epoch_num}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        model_path = os.path.join(save_dir, f"model_epoch_{epoch_num}.pth")
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, model_path)

        # Optionally save the best model
        if save_best and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, best_model_path)
            print(f"Best model saved at epoch {epoch_num} with Val Loss: {avg_val_loss:.4f}")
