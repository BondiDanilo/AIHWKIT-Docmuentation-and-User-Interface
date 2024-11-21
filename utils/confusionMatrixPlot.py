#this script creates a standardized confusion matrix plot over the KMNIST database

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

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

# Function to plot the confusion matrix
def confusionMatrixPlot1(model, dataloader, saveConfusionMatrix=None):
    # Set the model to evaluation mode (important for layers like dropout and batch norm)
    model.eval()
    
    # Lists to store all predictions and labels
    all_preds = []  # To store predicted labels
    all_labels = []  # To store true labels
    
    # Loop through the test dataset in the dataloader
    with torch.no_grad():  # Disable gradient computation (saves memory and computation)
        for X, y in dataloader:
            # Move data and labels to the appropriate device (CPU or GPU)
           
            # Get the model's predictions for the batch
            preds = model(X)
            
            # Get the index of the maximum prediction score for each sample in the batch
            # torch.max returns the value and the index, we only need the index (the predicted class)
            _, predicted = torch.max(preds, 1)
            
            # Save the predicted and true labels (move them to CPU if on GPU)
            all_preds.extend(predicted.numpy())  # Convert tensor to numpy array and add to the list
            all_labels.extend(y.numpy())  # Same for the true labels
    
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
                 annot_kws={"size": 11, 'weight': 'bold'},
                 linewidths=0.75, cbar=True,
                 linecolor='white'
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
                    ax.text(j + 0.5, i + 0.5, f'{data[i, j]:.1f}%', 
                            ha='center', va='center', color='white',
                                                    fontsize=11, fontweight='bold')
                    
    annotate_diagonal(cm_percentage, mask, ax)

    # Add title and axis labels to the plot
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label',fontsize=14)
    
    plt.savefig(saveConfusionMatrix, format='svg')

    # Display the plot
    plt.show()
