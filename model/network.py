import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        
        # Output Layer for Keypoints
        self.fc_keypoints = nn.Linear(in_features=512, out_features=8)  # 4 keypoints * 2 (x, y)
        
        # Dropout Layer for Regularization
        self.drop1 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # Convolutional Blocks with ReLU Activation and Max Pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Output: [Batch, 32, H/2, W/2]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Output: [Batch, 64, H/4, W/4]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Output: [Batch, 128, H/8, W/8]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Output: [Batch, 256, H/16, W/16]
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # Output: [Batch, 512, H/32, W/32]

        # Adaptive Average Pooling to Reduce Spatial Dimensions to 1x1
        x = F.adaptive_avg_pool2d(x, 1)  # Output: [Batch, 512, 1, 1]
        x = x.view(x.size(0), -1)        # Flatten: [Batch, 512]
        
        # Fully Connected Layers with ReLU Activation and Dropout
        x = F.relu(self.fc1(x))          # Output: [Batch, 1024]
        x = self.drop1(x)
        x = F.relu(self.fc2(x))          # Output: [Batch, 512]
        x = self.drop1(x)
        
        # Keypoints Output
        keypoints = self.fc_keypoints(x) # Output: [Batch, 8]
        
        # Return Only Keypoints
        return keypoints
