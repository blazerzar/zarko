
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.io import read_image
from torchvision import models

img_labels = pd.read_csv("20220619.csv")

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        
        tabular = self.img_labels.iloc[idx, 1:].values
        #date to hour
        tabular[0] = pd.to_datetime(tabular[0]).hour
        tabular = tabular.astype(np.float32)
        
        
        if idx >= len(self.img_labels) - 1:
            label = self.img_labels.iloc[idx, 2]
        else:
            label = self.img_labels.iloc[idx + 1, 2]

        image = image.float()
        tabular = torch.tensor(tabular, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, tabular, label


class Zarko(nn.Module):
    def __init__(self, cnn_features=20, tabular_features=10):
        super(Zarko, self).__init__()
        
        self.cnn_features = cnn_features
        self.tabular_features = tabular_features
        
        self.cnn = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        
        # Reduce the size of the final layer to 10/20
        # I have no concrete reason to do this, but I think it's a good idea
        # so the final prediction will not take in 1000 features from cnn and like 5 from tabular data
        self.cnn.classifier[3] = nn.Linear(self.cnn.classifier[3].in_features, self.cnn_features)
        
        self.fc1 = nn.Linear(self.cnn_features + self.tabular_features, 20)
        self.fc2 = nn.Linear(20, 1)
        
    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = data
        
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
model = Zarko(cnn_features=10, tabular_features=3)

data = CustomImageDataset("20220619.csv", "maribor_letalisce_20220619")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataloader = DataLoader(data, batch_size=8, shuffle=True)

n_epochs = 10
model.train()
for epoch in range(n_epochs):
    for i, (image, tabular, label) in enumerate(train_dataloader):
        pred = model(image, tabular)
        loss = F.mse_loss(pred, label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 16 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        