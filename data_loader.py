# Hasan Taha Bağcı 150210338
import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=64, augment=True): # Function to get CIFAR-10 data loaders
    """
    Prepares CIFAR-10 dataset loaders with specified transformations.
    Includes normalization and optional on-the-fly augmentations.
    """
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], # CIFAR-10 mean
                                     std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 std

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), # Random crop
            transforms.RandomHorizontalFlip(),                  # Random horizontal flip
            transforms.RandomRotation(10),                      # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color jitter
            transforms.ToTensor(),                              # Convert to tensor
            normalize,                                          # Normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),                              # Convert to tensor
            normalize,                                          # Normalize
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),                                  # Convert to tensor
        normalize,                                              # Normalize
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, # Load training data
                                                download=True, transform=train_transform)
    except Exception as e:
        print(f"Failed to download/load CIFAR10 training set. Trying with download=False (assuming already downloaded). Error: {e}")
        try:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=False, transform=train_transform)
        except Exception as final_e:
            print(f"Still failed to load CIFAR10 training set. Please check your internet connection or data directory. Error: {final_e}")
            return None, None, None


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, # Create training data loader
                                              shuffle=True, num_workers=2)

    try:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, # Load test data
                                               download=True, transform=test_transform)
    except Exception as e:
        print(f"Failed to download/load CIFAR10 test set. Trying with download=False. Error: {e}")
        try:
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=False, transform=test_transform)
        except Exception as final_e:
            print(f"Still failed to load CIFAR10 test set. Please check your internet connection or data directory. Error: {final_e}")
            return None, None, None

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, # Create test data loader
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', # CIFAR-10 classes
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

if __name__ == '__main__':
    # Example usage:
    train_loader, test_loader, class_names = get_cifar10_loaders(batch_size=4, augment=True)

    if train_loader and test_loader:
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        print(f"Classes: {class_names}")

        # Get some random training images
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"Example labels: {labels}")
    else:
        print("Failed to load data.")
