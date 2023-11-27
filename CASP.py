import torch
import torch.nn as nn
from models.resnet import ResNet18
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import random
import torchvision.transforms as transforms
import torchvision
import math

from torch.utils.data import Dataset
import pickle

    
soft_1 = nn.Softmax(dim=1)

def distribute_samples(probabilities, M):
    # Normalize the probabilities
    total_probability = sum(probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}

    # Calculate the number of samples for each class
    samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
    
    # Check if there's any discrepancy due to rounding and correct it
    discrepancy = M - sum(samples.values())
    
    # Adjust the number of samples in each class to ensure the total number of samples equals M
    for key in samples:
        if discrepancy == 0:
            break    # Stop adjusting if there's no discrepancy
        if discrepancy > 0:
            # If there are less samples than M, add a sample to the current class and decrease discrepancy
            samples[key] += 1
            discrepancy -= 1
        elif discrepancy < 0 and samples[key] > 0:
            # If there are more samples than M and the current class has samples, remove one and increase discrepancy
            samples[key] -= 1
            discrepancy += 1

    return samples    # Return the final classes distribution

    
def distribute_excess(lst):
    # Calculate the total excess value
    total_excess = sum(val - 500 for val in lst if val > 500)

    # Number of elements that are not greater than 500
    recipients = [i for i, val in enumerate(lst) if val < 500]

    num_recipients = len(recipients)

    # Calculate the average share and remainder
    avg_share, remainder = divmod(total_excess, num_recipients)

    lst = [val if val <= 500 else 500 for val in lst]
    
    # Distribute the average share
    for idx in recipients:
        lst[idx] += avg_share
    
    # Distribute the remainder
    for idx in recipients[:remainder]:
        lst[idx] += 1
    
    # Cap values greater than 500
    for i, val in enumerate(lst):
        if val > 500:
            return distribute_excess(lst)
            break

    return lst


def CASP_update(train_loader, Epoch, y_train, params_name):

    # Function CASP_update: Determine the Standard Deviation of Samples and Classes

    # Inputs:
    # train_loader: DataLoader object for training
    # Epoch: Number of training epochs
    # y_train: Ground truth labels for the training data
    # params_name: Input Parameters
    
    # Collect all unique classes from the training data
    unique_classes = set()
    for _, labels, indices_1 in train_loader:
        unique_classes.update(labels.numpy())
    
    # Set the device for computation and initialize the model
    device = "cuda"
    Surrogate_Model = ResNet18(len(unique_classes), params_name)
    Surrogate_Model = Surrogate_Model.to(device)

    # Define the loss function and optimizer
    criterion_ = nn.CrossEntropyLoss()
    optimizer_ = optim.SGD(Surrogate_Model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)

    # Define the learning rate scheduler
    scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=200)
    
    # Create mappings for class labels
    mapping = {value: index for index, value in enumerate(unique_classes)}
    reverse_mapping = {index: value for value, index in mapping.items()}

    # Initialize dictionaries to store confidence scores by class and epoch    
    confidence_by_class = {class_id: {epoch: [] for epoch in range(Epoch)} for class_id, __ in enumerate(unique_classes)}

    # Initialize a tensor to record confidence scores for samples across epochs
    confidence_by_sample = torch.zeros((Epoch, len(y_train)))

    # Training loop for the specified number of epochs
    for epoch_ in range(Epoch):
        print('\nEpoch: %d' % epoch_)
        Surrogate_Model.train()
        train_loss = 0
        correct = 0
        total = 0
        confidence_epoch = []

        # Loop over batches in the training data
        for batch_idx, (inputs, targets, indices_1) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Map targets to their corresponding indices
            targets = torch.tensor([mapping[val.item()] for val in targets]).to(device)

            # Zero the parameter gradients
            optimizer_.zero_grad()
            
            outputs = Surrogate_Model(inputs)
            soft_ = soft_1(outputs)
            confidence_batch = []
    
            # Accumulate confidences
            for i in range(targets.shape[0]):
                confidence_batch.append(soft_[i,targets[i]].item())
                
                # Update the dictionary with the confidence score for the current class for the current epoch
                confidence_by_class[targets[i].item()][epoch_].append(soft_[i, targets[i]].item())

            # Compute loss, perform backpropagation, and update model parameters
            loss = criterion_(outputs, targets)
            loss.backward()
            optimizer_.step()

            # Update training statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Record the confidence scores for samples in the corresponding tensor
            conf_tensor = torch.tensor(confidence_batch)
            confidence_by_sample[epoch_, indices_1] = conf_tensor
            
        print("Accuracy:", 100.*correct/total, ", and:", correct, "/", total, " ,loss:", train_loss/(batch_idx+1))

        # Update the learning rate
        scheduler_.step()

    # Calculate mean confidence by class
    mean_by_class = {class_id: {epoch: torch.mean(torch.tensor(confidences[epoch])) for epoch in confidences} for class_id, confidences in confidence_by_class.items()}

    # Calculate standard deviation of mean confidences by class
    std_of_means_by_class = {class_id: torch.std(torch.tensor([mean_by_class[class_id][epoch] for epoch in range(Epoch)])) for class_id, __ in enumerate(unique_classes)}
    
    # Compute mean and variability of confidences for each sample
    Confidence_mean = confidence_by_sample.mean(dim=0)
    Variability = confidence_by_sample.std(dim=0)

    # Outputs:
    # unique_classes: Set of unique classes in the dataset
    # mapping: Dictionary mapping class labels to indices
    # std_of_means_by_class: Standard deviation of mean confidence scores by class
    # Variability: Standard deviation of confidence scores across samples
    return unique_classes, mapping, std_of_means_by_class, Variability

def CASP_fill(unique_classes, mapping, std_of_means_by_class, Variability, train_dataset, y_train, buffer):

    # Inputs:
    # unique_classes: A set of unique class labels in the dataset
    # mapping: A dictionary mapping class labels to their indices
    # std_of_means_by_class: Standard deviation of means for each class
    # Variability: A tensor representing the variability of data points
    # train_dataset: The training dataset
    # y_train: The training labels
    # buffer: A buffer object containing data samples and labels

    # Initialize an empty list to store indices
    list_of_indices = []
    # Initialize a counter
    counter__ = 0
    # Iterate over each label in the buffer
    for i in range(buffer.buffer_label.shape[0]):
        # Check if the label is in the set of unique classes
        if buffer.buffer_label[i].item() in unique_classes:
            # Increment the counter and add the index to the list
            counter__ +=1
            list_of_indices.append(i)

    # Store the total count in top_n
    top_n = counter__

    # Sort indices based on the Confidence
    ##sorted_indices_1 = np.argsort(Confidence_mean.numpy())
    
    # Sort indices based on the variability
    sorted_indices_2 = np.argsort(Variability.numpy())
    


    ##top_indices_sorted = sorted_indices_1 #hard
    
    ##top_indices_sorted = sorted_indices_1[::-1] #simple

    # Descending order
    top_indices_sorted = sorted_indices_2[::-1] #challenging

    # Create a subset of the train dataset using the sorted indices
    subset_data = torch.utils.data.Subset(train_dataset, top_indices_sorted)
    # Create a DataLoader for the subset data
    trainloader_C = torch.utils.data.DataLoader(subset_data, batch_size=10, shuffle=False, num_workers=0)

    # Initialize lists to store images and labels
    images_list = []
    labels_list = []

    # Iterate over batches of images and labels from the DataLoader
    for images, labels, indices_1 in trainloader_C:
        # Append the images and labels to their respective lists
        images_list.append(images)
        labels_list.append(labels)

    # Concatenate all images and labels along the first dimension
    all_images = torch.cat(images_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Convert standard deviation of means by class to item form
    updated_std_of_means_by_class = {k: v.item() for k, v in std_of_means_by_class.items()}

    # Distribute samples based on the standard deviation
    dist = distribute_samples(updated_std_of_means_by_class, top_n)

    # Calculate the number of samples per class
    num_per_class = top_n//len(unique_classes)
    # Initialize a counter for each class
    counter_class = [0 for _ in range(len(unique_classes))]

    if len(y_train) == top_n:
        # Uniform distribution with adjustments for any remainder
        condition = [num_per_class for _ in range(len(unique_classes))]
        diff = top_n - num_per_class*len(unique_classes)
        for o in range(diff):
            condition[o] += 1
    else:
        # Distribution based on the class variability
        condition = [value for k, value in dist.items()]

    # Check if any class exceeds its allowed number of samples
    check_bound = len(y_train)/len(unique_classes)
    for i in range(len(condition)):
        if condition[i] > check_bound:
            # Redistribute the excess samples
            condition = distribute_excess(condition)
            break

    # Initialize new lists for adjusted images and labels
    images_list_ = []
    labels_list_ = []

    # Iterate over all_labels and select most challening images for each class based on the class variability
    for i in range(all_labels.shape[0]):
        if counter_class[mapping[all_labels[i].item()]] < condition[mapping[all_labels[i].item()]]:
            counter_class[mapping[all_labels[i].item()]] += 1
            labels_list_.append(all_labels[i])
            images_list_.append(all_images[i])
        if counter_class == condition:
            break

    # Stack the selected images and labels
    all_images_ = torch.stack(images_list_)
    all_labels_ = torch.stack(labels_list_)

    # Shuffle the data
    indices = torch.randperm(all_images_.size(0))
    shuffled_images = all_images_[indices]
    shuffled_labels = all_labels_[indices]

    # Update the buffer with the shuffled images and labels
    buffer.buffer_label[list_of_indices] = shuffled_labels.to("cuda")
    buffer.buffer_img[list_of_indices] = shuffled_images.to("cuda")
    
