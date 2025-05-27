# Hasan Taha Bağcı
# 150210338
# CIFAR-10 Data Loading and Preprocessing Utilities

import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=64, data_augmentation=True, num_workers=2): # Function to get CIFAR-10 data loaders
    """
    Prepares CIFAR-10 dataset loaders with specified transformations.
    Includes normalization and optional on-the-fly augmentations.
    """
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], # CIFAR-10 mean
                                     std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 std

    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),          # Random crop
            transforms.RandomHorizontalFlip(),             # Random horizontal flip
            transforms.RandomRotation(10),                 # Random rotation
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Random shifts (can be added)
            transforms.ToTensor(),                         # Convert image to PyTorch Tensor
            normalize,                                     # Normalize image
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, # Load training data
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, # Create training data loader
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, # Load test data
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, # Create test data loader
                                             shuffle=False, num_workers=num_workers)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # CIFAR-10 classes

    return trainloader, testloader

if __name__ == '__main__':
    # Example usage
    train_loader, test_loader = get_cifar10_loaders(batch_size=4, data_augmentation=True)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"Image batch shape: {images.shape}") # B, C, H, W
    print(f"Label batch shape: {labels.shape}")