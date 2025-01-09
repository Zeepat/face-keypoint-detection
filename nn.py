from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class LandmarkFaceAndSexNN(nn.Module):
    def __init__(self):
        super(LandmarkFaceAndSexNN, self).__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(8, 64),  # 8 inputs for the landmarks
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Branch for bounding box prediction
        self.bbox_fc = nn.Linear(32, 4)  # Predicts x, y, w, h
        
        # Branch for sex classification
        self.sex_fc = nn.Linear(32, 2)  # Predicts male (1) or female (0)
    
    def forward(self, x):
        shared_features = self.shared_fc(x)
        
        # Outputs
        bbox = self.bbox_fc(shared_features)  # Bounding box regression
        sex = self.sex_fc(shared_features)   # Sex classification
        
        return bbox, sex


class LandmarkFaceAndSexDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # Features: Landmark coordinates
        self.features = self.data[['LeftEye_x', 'LeftEye_y', 'RightEye_x', 'RightEye_y',
                                   'Nose_x', 'Nose_y', 'Mouth_x', 'Mouth_y']].values.astype(np.float32)
        self.features /= self.features.max(axis=0)  # Normalize landmarks
        
        # Targets: Bounding box (x, y, w, h) and sex (0/1)
        self.bboxes = self.data[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].values.astype(np.float32)
        self.labels = (self.data['sex'] == 'm').astype(np.int64).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.features[idx]
        bbox = self.bboxes[idx]
        label = self.labels[idx]
        return torch.tensor(x), torch.tensor(bbox), torch.tensor(label)



# Initialize model, loss functions, and optimizer
model = LandmarkFaceAndSexNN().cuda()
bbox_criterion = nn.SmoothL1Loss()  # For bounding box regression
sex_criterion = nn.CrossEntropyLoss()  # For sex classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create dataset and dataloader
dataset = LandmarkFaceAndSexDataset(csv_file='./aflw_face_landmarks.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    for batch_x, batch_bbox, batch_y in dataloader:
        batch_x, batch_bbox, batch_y = batch_x.cuda(), batch_bbox.cuda(), batch_y.cuda()  # Send to GPU
        
        optimizer.zero_grad()
        pred_bbox, pred_sex = model(batch_x)
        
        # Compute losses
        bbox_loss = bbox_criterion(pred_bbox, batch_bbox)
        sex_loss = sex_criterion(pred_sex, batch_y)
        total_loss = bbox_loss + sex_loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Total Loss: {total_loss.item()}, BBox Loss: {bbox_loss.item()}, Sex Loss: {sex_loss.item()}")
