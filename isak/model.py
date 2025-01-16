import torch.nn as nn
import torch.nn.functional as F

# https://github.com/nalbert9/Facial-Keypoint-Detection/blob/master/models.py

class nalbert_Net(nn.Module):

    def __init__(self):
        super(nalbert_Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 8)  # 4 keypoints istället för 64 i nalbert
        
        self.drop1 = nn.Dropout(p=0.25)
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = self.fc3(x)
        
        return x

class nalbert_Net_wBbox(nn.Module):

    def __init__(self):
        super(nalbert_Net_wBbox, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
        self.fc_keypoints = nn.Linear(512, 8)  # 4 keypoints * 2 (x, y)
        self.fc_bbox = nn.Linear(512, 4)      # Bounding box (x_min, y_min, x_max, y_max)
        
        self.drop1 = nn.Dropout(p=0.5)
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        
        keypoints = self.fc_keypoints(x)
        bbox = self.fc_bbox(x)
        
        return keypoints, bbox