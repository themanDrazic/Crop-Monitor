# Don't actually use this other than to grab some code snippets.
# computation intensive, most likely will not work on Raspberry Pi 3
import cv2
import VL53L1X
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define a simple CNN model for regression
class CornHeightCNN(nn.Module):
    # define your model here
    pass

# Custom dataset class
class CornDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_paths, lidar_measurement = self.data[idx]
        images = [cv2.imread(path) for path in image_paths]
        if self.transform:
            images = [self.transform(image) for image in images]
        return images, lidar_measurement

    def append(self, item):
        self.data.append(item)

# Initialize model, loss function, and optimizer
model = CornHeightCNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize dataset and data loader
dataset = CornDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Initialize LIDAR sensor
tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof.open()

# Initialize cameras
cameras = get_connected_cameras()

# Start continuous loop for data collection and training
while True:
    # Capture images and get LIDAR measurement
    images = capture_images(cameras)
    tof.start_ranging(1)
    lidar_measurement = tof.get_distance()
    tof.stop_ranging()

    # Add data to dataset and update data loader
    dataset.append((images, lidar_measurement))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Perform one step of training
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Delay for a set amount of time
    time.sleep(5)
