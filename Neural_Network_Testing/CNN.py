import torch
from torch import nn
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

import pandas as pd

import os

# Need to change this depending on your local directory of the dataset
# folder_dir = "C:/Users/61435/OneDrive/Documents/Personal/Convolutional Neural Network/SpiderManHerosVillainsDataSet/combinedDataSet"
folder_dir = "~/Documents/SpiderManHerosVillainsDataSet/combinedDataSet/"

transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),  # Rotate images randomly up to 30 degrees
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # Randomly crop and resize images
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Randomly adjust brightness and contrast
    transforms.GaussianBlur(kernel_size=3),  # Apply random Gaussian blur with kernel size 3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])]
)
    
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.rotation_angles = [0, 90, 180, 270]  # Rotation angles in degrees

        # Store the original image indices for each rotation
        self.rotated_indices = {0: [], 90: [], 180: [], 270: []}

        # Create rotated versions of images and store their indices
        for idx in range(len(self.img_labels)):
            img_name = self.img_labels.iloc[idx, 0]
            for angle in self.rotation_angles:
                self.rotated_indices[angle].append(len(self.img_labels))
                self.img_labels.loc[len(self.img_labels)] = [img_name, self.img_labels.iloc[idx, 1]]

    def __len__(self):
        return len(self.img_labels)

    def __padImage__(self, image, target_size):
        # Resize the image while preserving aspect ratio
        resized_image = image.resize(target_size, Image.LANCZOS)
        
        # Create a black canvas of the target size
        padded_image = Image.new("RGB", target_size, (0, 0, 0))
        
        # Paste the resized image onto the canvas
        padded_image.paste(resized_image, ((target_size[0] - resized_image.size[0]) // 2,
                                            (target_size[1] - resized_image.size[1]) // 2))
    
        return padded_image

    def __getitem__(self, idx):
        original_idx = idx % len(self.img_labels)  # Get the original index before rotation
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[original_idx, 0])
        image = ImageOps.grayscale(Image.open(img_path))
        
        # Rotate the image based on the rotation index
        rotation_angle = 0
        for angle, indices in self.rotated_indices.items():
            if original_idx in indices:
                rotation_angle = angle
                break
        image = image.rotate(rotation_angle, expand=True)  # Expand=True to preserve the image size
        
        # Apply padding to ensure consistent image size
        image = self.__padImage__(image, (256, 256))  # Resize images to 256x256

        label = self.img_labels.iloc[original_idx, 1]

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

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
print(f"Label: {label}")

# Hyperparameters
BATCH_SIZE = 12
HIDDEN_UNITS = 75
LEARNING_RATE = 0.006
EPOCHS = 1000

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
rows, cols = 6, 6
for i in range(1, rows * cols + 1):
    randomIdx = torch.randint(0, len(training_data), size=[1]).item()
    image, label = training_data[randomIdx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(image.squeeze().numpy(), cmap="gray")
    plt.title(classNames[label])
    plt.axis(False)
plt.show()

from CNN_model import SpideyDroneNet

model = SpideyDroneNet(1, HIDDEN_UNITS, len(classNames)).to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

trainLossLs = np.array([])
trainAccuracyLs = np.array([])
testLossLs = np.array([])
testAccuracyLs = np.array([])

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

    trainLoss /= len(train_dataloader) # train loss and acc per batch
    trainAcc /= len(train_dataloader)
    trainLossLs = np.append(trainLossLs, trainLoss.item())
    trainAccuracyLs = np.append(trainAccuracyLs, trainAcc)
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
        testLossLs = np.append(testLossLs, testLoss.item())
        testAccuracyLs = np.append(testAccuracyLs, testAcc)
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

def plot_loss(train_Loss, test_Loss):
    epochNum = range(1, len(train_Loss) + 1)
    plt.plot(epochNum, train_Loss, 'b', label='Training Loss')
    plt.plot(epochNum, test_Loss, 'g', label='Test Loss')
    plt.title("Training and Test Loss Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_acc(train_Acc, test_Acc):
    epochNum = range(1, len(train_Acc) + 1)
    plt.plot(epochNum, train_Acc, 'b', label='Training Accuracy')
    plt.plot(epochNum, test_Acc, 'g', label='Test Accuracy')
    plt.title("Training and Test Accuracy Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

plot_loss(trainLossLs, testLossLs)
plot_acc(trainAccuracyLs, testAccuracyLs)

torch.save(model.state_dict(), 'model_pytorch.pt')