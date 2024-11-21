#%% import libraries and packages

#this code trains a DNN over KMNIST dataset. Adapted from the
#pytorch notebook tutorial series
#it improves from the first version on accuracy, and code

import os
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
from torch.nn.functional import mse_loss
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# %% download training data 
dataPath_mio = '/Users/iiporza/ecramai/data'
#every dataset contains two arguments, transform and target transform to modify samples and labels

# Download training data from open datasets.
training_data = datasets.KMNIST(
    #root="data",
    dataPath_mio,
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.KMNIST(
    #root="data",
    dataPath_mio,
    train=False,
    download=True,
    transform=ToTensor(),
)

# %% 
#pass dataset as an argument to dataloader. This wraps an iterable over our dataset.
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#this is helpful for debugging and checking the structure of the data before training
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") #the shape will be batch size, channels, height, width
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#for the X
#batch size is given by the dataloader function 
#channels are 1 since the image is in greyscale
#the images are 28x28 so height, width

#for the y
#the y are the labels. batch size is 64, while the label is a number integer

#the break stops the loop after the first batch of the test dataset, and is useful to inspect the shapes before running the full training process

# %% interacting and visualizing the dataset

#the data can be indexed manually with training_data[index]

import matplotlib.pyplot as plt

labels_map = {
    0: "o",
    1: "ki",
    2: "su",
    3: "tsu",
    4: "na",
    5: "ha",
    6: "ma",
    7: "ya",
    8: "re",
    9: "wo",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() #generate random number
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %% creating the model

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
#the neural network is defined by subclassing nn.Module,
#  it is initialized in __init__. Every subclass implements the operations in the forward method

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x) #converts input into contiguous array. It maintains the minibatch dimension dim = 0
        logits = self.linear_relu_stack(x)
        return logits

#it creates an instance of NeuralNetwork, and moves it to device
model = NeuralNetwork().to(device)
print(model) 

#this approach uses nn.sequential within the definition of a class. The data is passed through all the modules in 
#the same order as it is defined. 
#to use the model, we pass it the input data, which executes the model's "forward".
#do not call directly the model

#by using the model's parameters() or named_parameters() methods, 

# %% Optimizing the model parameters

#to train a model, a loss function and an optimizer are needed. 

loss_fn = nn.NLLLoss() #the other example uses nn.NLLLoss
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

#then we define a training loop to make predictions on the input data, and then adjust

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # Set the model to train mode (important for layers like dropout and batch norm)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #In this case, flatten operation is perfor
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y) # Compute the loss by comparing the predictions with the true labels

        # Backpropagation
        loss.backward() # Backpropagation: Compute the gradients of the loss with respect to the model's parameters
        optimizer.step()  # Update the model's weights based on the computed gradients
        optimizer.zero_grad()  # Zero the gradients for the next step (otherwise gradients will accumulate)

        # Print the loss every 100 batches
        if batch % 100 == 0: 
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader) #number of batches in the dataloader
    model.eval() # Set the model to evaluation mode (important for layers like dropout and batch norm)
    test_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy_values.append(100*correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %% 
accuracy_values = [] #array to store accuracy values
epochs = 100
for t in tqdm(range(epochs),  desc="Training Progress"):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# %%
#Plot the accuracy as a function of the epoch
savePath = '/Users/iiporza/desktop/ai ecram work'
saveConfusionMatrix = savePath + '/accuracy1.svg'


plt.plot(range(epochs), accuracy_values, label="Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)

plt.savefig(saveConfusionMatrix, format='svg')

plt.show()

# %% create the confusion matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

savePath = '/Users/iiporza/desktop/ai ecram work'
saveConfusionMatrix = savePath + '/confusionMatrix1.svg'

# Function to plot the confusion matrix
def plot_confusion_matrix(model, dataloader, device):
    # Set the model to evaluation mode (important for layers like dropout and batch norm)
    model.eval()
    
    # Lists to store all predictions and labels
    all_preds = []  # To store predicted labels
    all_labels = []  # To store true labels
    
    # Loop through the test dataset in the dataloader
    with torch.no_grad():  # Disable gradient computation (saves memory and computation)
        for X, y in dataloader:
            # Move data and labels to the appropriate device (CPU or GPU)
            X, y = X.to(device), y.to(device)
            
            # Get the model's predictions for the batch
            preds = model(X)
            
            # Get the index of the maximum prediction score for each sample in the batch
            # torch.max returns the value and the index, we only need the index (the predicted class)
            _, predicted = torch.max(preds, 1)
            
            # Save the predicted and true labels (move them to CPU if on GPU)
            all_preds.extend(predicted.cpu().numpy())  # Convert tensor to numpy array and add to the list
            all_labels.extend(y.cpu().numpy())  # Same for the true labels
    
    # Compute the confusion matrix using sklearn's confusion_matrix function
    cm = confusion_matrix(all_labels, all_preds)
    
    # Convert the counts to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Handle division by zero or NaN values by filling NaNs with zero
    cm_percentage = np.nan_to_num(cm_percentage)
    
    #create a mask for diagonal elements
    mask = np.zeros_like(cm_percentage, dtype='bool')
    np.fill_diagonal(mask,True) 

    # Plot the confusion matrix using seaborn's heatmap function
    plt.figure(figsize=(10, 7))  # Set the size of the plot
    ax = sns.heatmap(cm_percentage, annot=False, fmt='.1f', cmap='BuPu',
                 xticklabels=[labels_map[i] for i in range(10)], 
                 yticklabels=[labels_map[i] for i in range(10)],
                 cbar_kws={'label': 'Percentage'},
                 annot_kws={"size": 12, 'weight': 'bold'},
                 linewidths=0.5, cbar=True,
                 linecolor='black'
                 )
    # annot=True: Display the numerical value in each cell
    # fmt='d': Format as integers (since it's a confusion matrix with counts)
    # cmap='Blues': Use the blue color map for better visualization
    # xticklabels and yticklabels: Label the axes with class numbers (0-9 for KMNIST)
     # Annotate only the diagonal elements
   
    # Custom annotation function
    def annotate_diagonal(data, mask, ax):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if mask[i, j]:
                    ax.text(j + 0.5, i + 0.5, f'{data[i, j]:.1f}', 
                            ha='center', va='center', color='white',
                                                    fontsize=12, fontweight='bold')
                    
    annotate_diagonal(cm_percentage, mask, ax)

    # Add title and axis labels to the plot
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.savefig(saveConfusionMatrix, format='svg')

    # Display the plot
    plt.show()

# Example usage: Call the function with your trained model, test dataloader, and device
plot_confusion_matrix(model, test_dataloader, device)


# %%
