from torch.utils.data import Dataset, DataLoader, random_split
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
        try:
            img_name = os.path.join(self.root_dir, self.annotations["file_id"][idx])
            image = plt.imread(img_name)
            
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[:, :, :3]
            elif image.shape[-1] != 3:
                raise ValueError(f"Unexpected number of channels: {image.shape[-1]} in {img_name}")

            landmarks = self.annotations.iloc[idx, 2:-2].values
            landmarks = landmarks.astype('float').reshape(-1, 2)

            if self.transform:
                image = self.transform(image)

            h, w = image.shape[1:]
            landmarks[:, 0] *= (224 / w)
            landmarks[:, 1] *= (224 / h)

            return image, torch.tensor(landmarks, dtype=torch.float32)

        except Exception as e:
            print(f"Skipping {idx} due to error: {e}")
            return None
    

# dataset = FaceKeypointDataset(annotations, 'data/training/', transform=transform)

def train_test_split(dataset, train_size=0.8, val_size=0.1, batch_size=128):
    total_size = len(dataset)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

----------------------------------------------------------------------------------------------------------------------------
def train(model, criterion, optimizer, train_loader, val_loader, epochs=50, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    for epoch in range(10):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            if inputs is None:
                continue
            
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.view(labels.size(0), -1)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            # print(f"outputs shape: {outputs.shape}")
            # print(f"labels shape: {labels.shape}")

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10}")
                running_loss = 0.0