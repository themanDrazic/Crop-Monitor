# Use this after collecting data with the "create_dataset_for_ML_model.py" 
# program for collecting data
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define a simple CNN model for regression
class CornHeightCNN(nn.Module):
    # define your model here
    pass

# Custom dataset class
class CornDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_paths = self.data_frame.iloc[idx, 0]
        images = [Image.open(path) for path in image_paths.split(',')]
        if self.transform:
            images = [self.transform(image) for image in images]
        lidar_measurement = self.data_frame.iloc[idx, 1]
        weather = self.data_frame.iloc[idx, 2]
        temp = self.data_frame.iloc[idx, 3]
        time_of_day = self.data_frame.iloc[idx, 4]
        # Convert non-image data to tensor
        additional_data = torch.tensor([weather, temp, time_of_day])
        return images, additional_data, lidar_measurement

# Function to train the model
def train_model(model, criterion, optimizer, dataloader, num_epochs=25):
    for epoch in range(num_epochs):  
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, additional_data, labels = data
            optimizer.zero_grad()
            # Adjust model to accept additional data
            outputs = model(inputs, additional_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Loss: {:.4f}'.format(running_loss / len(dataloader)))

    print('Finished Training')

# Training
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
dataset = CornDataset(csv_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
model = CornHeightCNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_model(model, criterion, optimizer, dataloader)
