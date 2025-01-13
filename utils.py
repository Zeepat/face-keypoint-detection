from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from PIL import Image, ImageFile

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FaceKeypointDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None):
        self.annotations = annotations  # pandas dataframe
        self.root_dir = root_dir        # path to images
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
            landmarks = landmarks.astype('float').reshape(-1, 2)

            if self.transform:
                image = self.transform(img)
            else:
                # If no transform is provided, convert PIL image to tensor
                image = transforms.ToTensor()(img)

            h, w = image.shape[1:]
            landmarks[:, 0] *= (224 / w)
            landmarks[:, 1] *= (224 / h)

            # Bounding Boxes
            bboxes_cols = ['X_box', 'Y_box', 'W_box', 'H_box']
            bboxes = self.annotations.loc[idx, bboxes_cols].values
            bboxes = bboxes.astype('float').reshape(-1, 4)
            bboxes[:, 0] *= (224 / w)
            bboxes[:, 1] *= (224 / h)
            bboxes[:, 2] *= (224 / w)
            bboxes[:, 3] *= (224 / h)

            return image, torch.tensor(landmarks, dtype=torch.float32), torch.tensor(bboxes, dtype=torch.float32)

        except Exception as e:
            print(f"Skipping {idx} due to error: {e}")
            return None

def skip_none_collate_fn(batch):
    # Filter out any items that are None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None
    images, landmarks, bboxes = zip(*batch)
    images = default_collate(images)
    landmarks = default_collate(landmarks)
    bboxes = default_collate(bboxes)
    return images, landmarks, bboxes

def train_test_split(dataset, train_size=0.8, val_size=0.1, batch_size=32):
    total_size = len(dataset)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=skip_none_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=skip_none_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=skip_none_collate_fn)
    
    return train_loader, val_loader, test_loader

def train(model, criterion_keypoints, criterion_bbox, optimizer, train_loader, val_loader, epochs=50, device='cuda', bbox_weight=1.0):
    model.to(device)
    
    for epoch in range(epochs):
        epoch_num = epoch + 1
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        
        # Initialize tqdm for training batches
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{epochs} - Training", leave=False)
        
        for batch in train_bar:
            if batch[0] is None:
                continue  # Skip batches where all items were None
            images, landmarks, bboxes = batch
            images, landmarks, bboxes = images.to(device), landmarks.to(device), bboxes.to(device)
            
            optimizer.zero_grad()
            keypoints_pred, bboxes_pred = model(images)
            
            # Compute losses
            loss_keypoints = criterion_keypoints(keypoints_pred, landmarks.view(-1, 8))
            loss_bbox = criterion_bbox(bboxes_pred, bboxes.view(-1, 4))
            loss = loss_keypoints + bbox_weight * loss_bbox  # Combine losses
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update tqdm description with current loss
            train_bar.set_postfix({'Batch Loss': loss.item()})
        
        # Validation phase with tqdm
        model.eval()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch_num}/{epochs} - Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                if batch[0] is None:
                    continue  # Skip batches where all items were None
                images, landmarks, bboxes = batch
                images, landmarks, bboxes = images.to(device), landmarks.to(device), bboxes.to(device)
                
                keypoints_pred, bboxes_pred = model(images)
                
                loss_keypoints = criterion_keypoints(keypoints_pred, landmarks.view(-1, 8))
                loss_bbox = criterion_bbox(bboxes_pred, bboxes.view(-1, 4))
                loss = loss_keypoints + bbox_weight * loss_bbox
                
                val_loss += loss.item()
                
                # Update tqdm description with current validation loss
                val_bar.set_postfix({'Val Loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch_num}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
