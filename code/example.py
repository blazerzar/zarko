import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = models.inception_v3(pretrained=False, aux_logits=False)
        self.cnn.fc = nn.Linear(
            self.cnn.fc.in_features, 20)
        
        self.fc1 = nn.Linear(20 + 10, 60)
        self.fc2 = nn.Linear(60, 1)
        
    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = data
        
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        

model = MyModel()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)    
if device.type == 'cuda':
    print("Number of cuda devices: ", torch.cuda.device_count())

batch_size = 2
image = torch.randn(batch_size, 3, 299, 299)
data = torch.randn(batch_size, 10)

image = image.to(device)
data = data.to(device)
model = model.to(device)

output = model(image, data)
print("Output shape: ", output.shape)

