import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from get_data import Get_data
import albumentations as A
from Custom_Dataset import CustomDataset
from DataLoader import Loaders

#load the data
train_path = "../data/training.csv"
test_path = "../data/test.csv"

#transform
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5, limit=10),
        A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=10)
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
)

#loaders
loaders = Loaders(train_path, test_path, transform)
train_loader = loaders.get_loader('train', batch_size=32)
test_loader = loaders.get_loader('test', batch_size=32)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#load pretrained ResNet34
import torchvision
model = torchvision.models.resnet34(pretrained=True)
#input channel is 1
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#output layer gives 30 keypoints
model.fc = nn.Linear(512, 30)
model.to(device)

#loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#train the model
epochs = 500
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.view(-1, 30).to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
    #for every 100 epochs, save checkpoint
    if (epoch+1) % 100 == 0:
        torch.save(model.state_dict(), './checkpoint_'+str(epoch+1)+'.pth')
