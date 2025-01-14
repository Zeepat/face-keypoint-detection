import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
        # Separate heads for keypoints and bounding box
        self.fc_keypoints = nn.Linear(512, 8)  # 4 keypoints * 2 (x, y)
        self.fc_bbox = nn.Linear(512, 4)      # Bounding box (x_min, y_min, x_max, y_max)
        
        self.drop1 = nn.Dropout(p=0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        
        # Separate outputs
        keypoints = self.fc_keypoints(x)
        bbox = self.fc_bbox(x)
        
        return keypoints, bbox
    
def _get_conv_output_shape(image_size):
    out_size = image_size
    for _ in range(4):  # Four convolutional layers followed by pooling
        out_size = (out_size - 1) // 2 + 1
    return out_size

class FacialLandmarkAndBBoxModel(nn.Module):
    def __init__(self):
        super(FacialLandmarkAndBBoxModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the flattened feature map after pooling

        conv_out_size = _get_conv_output_shape(224)
        
        # Fully connected layers for keypoint and bounding box prediction
        self.fc1 = nn.Linear(512 * conv_out_size * conv_out_size, 1024)
        self.fc_keypoints = nn.Linear(1024, 8)     # 4 keypoints * 2 (x, y)
        self.fc_bbox = nn.Linear(1024, 4)          # Bounding box (x_min, y_min, x_max, y_max)

    
    def forward(self, x):
        # Apply convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the feature map
        conv_out_size = _get_conv_output_shape(224)
        x = x.view(-1, 512 * conv_out_size * conv_out_size)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        
        keypoints = self.fc_keypoints(x)
        bbox = self.fc_bbox(x)

        return keypoints, bbox

# Example usage:
# model = FacialLandmarkAndBBoxModel()
# input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image
# keypoints_output, bbox_output = model(input_tensor)
# print(f"Keypoints output shape: {keypoints_output.shape}")
# print(f"BBox output shape: {bbox_output.shape}")