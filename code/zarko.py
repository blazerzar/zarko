import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
from torchvision.io import read_image
from torchvision import models
from transformers import pipeline
from sklearn.model_selection import train_test_split

import comet_ml

print('Imported')

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, use_detection=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.use_detection = use_detection
        if self.use_detection:
            comet_ml.init(anonymous=True, project_name="3: OWL-ViT + SAM")
            exp = comet_ml.Experiment()
            logged_artifact = exp.get_artifact("L3-data", "anmorgan24")
            OWL_checkpoint = "google/owlvit-base-patch32"
            self.detector = pipeline(
                model= OWL_checkpoint,
                task="zero-shot-object-detection"
            )

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        label = self.img_labels.iloc[idx, 2]
        image = image.float()

        if self.use_detection:
            image_pil = Image.open("images.jpeg")
            raw_image = image_pil.convert("RGB")
            text_prompt = "dog"
            output = self.detector(
                raw_image,
                candidate_labels = [text_prompt]
            )
            bbox = output[0]["box"]
            left = bbox["xmin"]
            top = bbox["ymin"]
            right = bbox["xmax"]
            bottom = bbox["ymax"]
            image = image[:, top:bottom, left:right]
            image = image.unsqueeze(0)

        train_transformations = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        image = train_transformations(image)

        label = torch.tensor(label, dtype=torch.float32)

        return image, label

class Zarko(nn.Module):
    def __init__(self, cnn_features=20, tabular_features=10):
        super(Zarko, self).__init__()

        self.cnn_features = cnn_features
        # self.tabular_features = tabular_features

        self.cnn = models.mobilenet_v3_small(pretrained=True)

        # Reduce the size of the final layer to 10/20
        # I have no concrete reason to do this, but I think it's a good idea
        # so the final prediction will not take in 1000 features from cnn and like 5 from tabular data
        self.cnn.classifier[3] = nn.Linear(self.cnn.classifier[3].in_features, self.cnn_features)

        # self.fc1 = nn.Linear(self.cnn_features + self.tabular_features, 20)
        self.fc1 = nn.Linear(self.cnn_features, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, image):
        x1 = self.cnn(image)
        # x2 = data

        # x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x1))
        x = self.fc2(x)
        return x

def train_zarko():
    model = Zarko(cnn_features=10, tabular_features=3).to('cuda')

    data = CustomImageDataset("maribor_letalisce_20220619_all.csv", "../data_maribor_airport/")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # remove truncated images
    cleaned_data = []
    for i in range(len(data)):
        try:
            image, label = data.__getitem__(i)
            cleaned_data.append((image, label))
        except Exception as e:
            print(e)
            continue
    print("Before", len(data), "After", len(cleaned_data))
    
    train_data, test_data = train_test_split(cleaned_data, test_size=0.2)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

    n_epochs = 50
    model.to(torch.device('cuda'))
    model.train()
    loss_fn = nn.L1Loss()

    for epoch in range(n_epochs):
        full_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        num_batches = 0
        try:
            for i, (image, label) in enumerate(train_dataloader):
                num_batches += 1
                image = image.to(torch.device('cuda'))
                label = label.to(torch.device('cuda'))
                optimizer.zero_grad()

                pred = model(image)

                loss = loss_fn(pred.flatten(), label)
                full_loss += loss
                loss.backward()
                optimizer.step()
                
            print("Epoch:", epoch, "Loss:", full_loss.item() / num_batches)
        except Exception as e:
            print(e)

    # test the model
    test_data = DataLoader(test_data, batch_size=64, shuffle=True)
    model.eval()
    with torch.no_grad():
        full_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        num_batches = 0
        for i, (image, label) in enumerate(test_data):
            num_batches += 1
            image = image.to(torch.device('cuda'))
            label = label.to(torch.device('cuda'))

            pred = model(image)

            loss = loss_fn(pred.flatten(), label)
            full_loss += loss

        print("Test loss:", full_loss.item() / num_batches)

    # save the model
    torch.save(model.state_dict(), "model.pth") 

if __name__ == "__main__":
    train_zarko()
    print('Done')
