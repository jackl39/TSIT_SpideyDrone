import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import numpy as np
import pandas as pd

from PIL import Image, ImageOps

import os
import pandas as pd
from torchvision.io import read_image
import cv2

folder_dir = "/home/jack/Documents/SpiderManHerosVillainsDataSet/combinedDataSet"

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __padImage__(self, image, target_size):
        # Resize the image while preserving aspect ratio
        resized_image = image.resize(target_size, Image.ANTIALIAS)
        
        # Create a black canvas of the target size
        padded_image = Image.new("RGB", target_size, (0, 0, 0))
        
        # Paste the resized image onto the canvas
        padded_image.paste(resized_image, ((target_size[0] - resized_image.size[0]) // 2,
                                            (target_size[1] - resized_image.size[1]) // 2))
    
        return padded_image
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = ImageOps.grayscale(Image.open(img_path))
        # image = self.__padImage__(image, 2967, 2400)
        image = self.__padImage__(image, (256, 256))  # Resize images to 256x256
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
# # Load the dataset
dataset = CustomImageDataset(annotations_file=folder_dir+'/annotations_clean.txt', img_dir=folder_dir, transform=transform)

# # Split the dataset into training and test dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

# Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on {}".format(device))

# # Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
print(f"Label: {label}")
# plt.imshow(train_features[0].permute(1, 2, 0).numpy(), cmap="gray")

# Hyperparameters
BATCH_SIZE = 20
HIDDEN_UNITS = 60
LEARNING_RATE = 0.05
EPOCHS = 15

def calculateAcc(yTrue, yPred):
    correct = torch.eq(yTrue, yPred).sum().item()
    acc = (correct/len(yPred)) * 100
    return acc

classNames = ["", "docOc", "electro", "greenGoblin", "theLizard", "venom"]

# Visualize the first piece of data
image, label = training_data[0]

# Visualize sample images
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    randomIdx = torch.randint(0, len(training_data), size=[1]).item()
    image, label = training_data[randomIdx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(image.squeeze().numpy(), cmap="gray")
    plt.title(classNames[label])
    plt.axis(False)
plt.show()


class KhitNet(nn.Module):
    def __init__(self, inputShape: int, hiddenUnits: int, outputShape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=inputShape,
                      out_channels=hiddenUnits,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnits,
                      out_channels=hiddenUnits,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(hiddenUnits, hiddenUnits, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hiddenUnits, hiddenUnits, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenUnits * 64 * 64,
                      out_features=outputShape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        # Uncomment this when running for the first time to determine linear layer's size
        # print(x.shape) 
        x = self.classifier(x)
        return x
    
model = KhitNet(1, HIDDEN_UNITS, len(classNames)).to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):

    print("Epoch: {}\n---------".format(epoch))

    # Put model into training mode
    model.train()

    # Keep track of running loss and accuracy
    trainLoss = 0
    trainAcc = 0

    for batch, (X, y) in enumerate(train_dataloader):

        # 0. Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward Pass
        yPred = model(X)

        # 2. Calculate Loss
        loss = lossFunction(yPred, y)
        trainLoss += loss
        trainAcc += calculateAcc(y, yPred.argmax(dim=1)) # Go from logits to labels
        
        # 3. Zero out gradients in optimizer
        optimizer.zero_grad()

        # 4. Backpropagation (Calculate the gradients)
        loss.backward()

        # 5. Gradient Descent (Minimize loss and update parameters)
        optimizer.step()

    trainLoss /= len(train_dataloader)
    trainAcc /= len(train_dataloader)
    print("Train Loss = {:.3f} Train Accuracy = {:.2f}%".format(trainLoss, trainAcc))

    # Testing
    model.eval()
    testLoss = 0
    testAcc = 0

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            testPred = model(X)
            testLoss += lossFunction(testPred, y)
            testAcc += calculateAcc(y, testPred.argmax(dim=1))

        testLoss /= len(test_dataloader)
        testAcc /= len(test_dataloader)
        print("Test Loss = {:.3f} Test Accuracy = {:.2f}%".format(testLoss, testAcc))

testSamples = []
testLabels = []
for sample, label in random.sample(list(test_data), k=9):
    testSamples.append(sample)
    testLabels.append(label)

predProbs = []
model.eval()
with torch.inference_mode():
    for sample in testSamples:
        sample = torch.unsqueeze(sample, dim=0).to(device)

        # Forward pass to get raw logits
        predLogit = model(sample)

        # Get prediction probability. Logits -> probability
        predProb = torch.softmax(predLogit.squeeze(), dim=0)

        # Matplotlib does not work with GPU
        predProbs.append(predProb.cpu())

predProbs = torch.stack(predProbs)
predClasses = predProbs.argmax(dim=1)

plt.figure(figsize=(9,9))
rows = 3
cols = 3

for i, sample in enumerate(testSamples):
    plt.subplot(rows, cols, i+1)
    plt.imshow(sample.squeeze(), cmap='gray')
    predLabel = classNames[predClasses[i]]
    truthLabel = classNames[testLabels[i]]
    title = "Pred: {} | Truth: {}".format(predLabel, truthLabel)

    if predLabel == truthLabel:
        plt.title(title, fontsize=10, c='g')
    else:
        plt.title(title, fontsize=10, c='r')
    plt.axis(False)
plt.show()

