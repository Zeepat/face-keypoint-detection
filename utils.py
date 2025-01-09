from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FaceKeypointDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None):
        
        self.annotations = annotations # pandas dataframe
        self.root_dir = root_dir # path till bilderna
        self.transform = transform # transform grejen ovan

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations["file_id"][idx]) # Den synkar bilderna med bilden som nämns i annoteringen
        image = plt.imread(img_name)
        
        if len(image.shape) == 2: # Denna gör om B&W till RGB (alltså bara 2 till 3 i den dimensionen)
            image = np.stack([image] * 3, axis=-1)

        landmarks = self.annotations.iloc[idx, 2:10].values # Tar ut keypointsen från annoteringarna
        landmarks = landmarks.astype('float').reshape(-1, 2) # Osäker på vad den gör

        if self.transform:
            image = self.transform(image)
            
        h, w = image.shape[1:] # Dessa rader syncar keypointsen så de funkar till de nya bilddimmensionerna
        landmarks[:, 0] *= (224 / w)
        landmarks[:, 1] *= (224 / h)

        return image, torch.tensor(landmarks, dtype=torch.float32)
    

# dataset = FaceKeypointDataset(annotations, 'data/training/', transform=transform)