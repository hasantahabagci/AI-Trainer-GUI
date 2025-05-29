# Hasan Taha Bağcı 150210338
# core/data_loader.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465) # Mean for CIFAR-10 normalization
CIFAR10_STD = (0.2023, 0.1994, 0.2010) # Std for CIFAR-10 normalization

def get_cifar10_loaders(batch_size=64, data_dir='./data', augment=True, num_workers=2, pin_memory=True): # Loads CIFAR-10 train and test dataloaders
    """
    Prepares the CIFAR-10 dataset with appropriate transformations and creates DataLoaders.

    Args:
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory to download/load CIFAR-10 data.
        augment (bool): Whether to apply data augmentation to the training set.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),          
            transforms.RandomHorizontalFlip(),             
            transforms.RandomRotation(15),                  
            transforms.ToTensor(),                         
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD) 
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    # Transformations for the test/validation set (only normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader, CIFAR10_CLASSES

def imshow(img_tensor, title=None): 
    """
    Displays an image tensor. Unnormalizes if normalized.
    """
    img_tensor = img_tensor / 2 + 0.5 
    
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img_tensor = img_tensor * std + mean # Unnormalize
    
    npimg = img_tensor.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # Convert from CxHxW to HxWxC
    if title:
        plt.title(title)
    plt.axis('off') # Turn off axis numbers and ticks

if __name__ == '__main__':
    print("Testing CIFAR-10 Data Loader...")
    
    # Get data loaders
    trainloader, testloader, classes = get_cifar10_loaders(batch_size=4, augment=True)
    print(f"Number of training batches: {len(trainloader)}")
    print(f"Number of test batches: {len(testloader)}")
    print(f"Classes: {classes}")

    # Get some random training images
    dataiter = iter(trainloader)
    try:
        images, labels = next(dataiter)
    except StopIteration:
        print("Could not fetch a batch from trainloader. Is the dataset empty or batch_size too large?")
        images, labels = None, None

    if images is not None and labels is not None:
        # Show images
        print('Labels: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(images.size(0))))
        
        # Create a figure to show images
        fig = plt.figure(figsize=(8, 2)) # Adjust figure size as needed
        imshow(torchvision.utils.make_grid(images))
        plt.suptitle("Sample Training Images (Augmented & Normalized)", fontsize=14)
        plt.show()

        # Test without augmentation to see original normalized images
        trainloader_no_aug, _, _ = get_cifar10_loaders(batch_size=4, augment=False)
        dataiter_no_aug = iter(trainloader_no_aug)
        try:
            images_no_aug, labels_no_aug = next(dataiter_no_aug)
        except StopIteration:
            images_no_aug, labels_no_aug = None, None
        
        if images_no_aug is not None:
            fig_no_aug = plt.figure(figsize=(8, 2))
            imshow(torchvision.utils.make_grid(images_no_aug))
            plt.suptitle("Sample Training Images (Normalized Only)", fontsize=14)
            plt.show()

    print("\nTesting Test Loader...")
    dataiter_test = iter(testloader)
    try:
        images_test, labels_test = next(dataiter_test)
    except StopIteration:
        images_test, labels_test = None, None

    if images_test is not None and labels_test is not None:
        print('Test Labels: ', ' '.join(f'{classes[labels_test[j]]:5s}' for j in range(images_test.size(0))))
        fig_test = plt.figure(figsize=(8,2))
        imshow(torchvision.utils.make_grid(images_test))
        plt.suptitle("Sample Test Images (Normalized)", fontsize=14)
        plt.show()
