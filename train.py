# Hasan Taha Bağcı 150210338
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_cifar10_loaders
from models import CustomCNN, get_resnet50, get_vgg16
import argparse
import json
import os
import time
import matplotlib.pyplot as plt

def train_model(model_name, epochs, lr, batch_size, augment, results_dir, run_id): # Main training function
    """
    Trains a specified CNN model on CIFAR-10.
    Logs metrics and saves plots.
    """
    print(f"Training Configuration:") # Print config
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Data Augmentation: {augment}")
    print(f"  Results Directory: {results_dir}")
    print(f"  Run ID: {run_id}")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True) # Create results directory
    log_file_path = os.path.join(results_dir, f"{run_id}_metrics.jsonl") # Path for metrics log

    # Setup device (MPS for Mac M-series, CUDA for Nvidia, else CPU)
    if torch.backends.mps.is_available(): # Check for Apple Silicon MPS
        device = torch.device("mps")
    elif torch.cuda.is_available(): # Check for CUDA
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") # Default to CPU
    print(f"Using device: {device}") # Print device being used

    # Load data
    print("Loading CIFAR-10 dataset...") # Log data loading
    trainloader, testloader, classes = get_cifar10_loaders(batch_size=batch_size, augment=augment)
    if trainloader is None or testloader is None:
        error_msg = "Failed to load dataset. Exiting."
        print(json.dumps({"type": "error", "message": error_msg}))
        return

    # Initialize model
    print(f"Initializing model: {model_name}...") # Log model initialization
    if model_name == 'CustomCNN':
        model = CustomCNN(num_classes=len(classes))
    elif model_name == 'ResNet50':
        model = get_resnet50(num_classes=len(classes), pretrained=True) # Use pretrained weights
    elif model_name == 'VGG16':
        model = get_vgg16(num_classes=len(classes), pretrained=True) # Use pretrained weights
    else:
        error_msg = f"Unknown model: {model_name}"
        print(json.dumps({"type": "error", "message": error_msg}))
        return
    model.to(device) # Move model to device

    criterion = nn.CrossEntropyLoss() # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # Define optimizer
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Optional learning rate scheduler

    history = { # Dictionary to store training history
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("Starting training...") # Log start of training
    for epoch in range(epochs): # Loop over the dataset multiple times
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        epoch_start_time = time.time() # Time epoch start

        for i, data in enumerate(trainloader, 0): # Loop through training data
            inputs, labels = data[0].to(device), data[1].to(device) # Get inputs and labels to device

            optimizer.zero_grad() # Zero the parameter gradients

            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Optimize

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0: # Print progress every 100 mini-batches
                batch_log = {
                    "type": "log",
                    "message": f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(trainloader)}], Train Loss: {loss.item():.4f}"
                }
                print(json.dumps(batch_log)) # Print log as JSON

        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # No gradients needed for validation
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(testloader)
        epoch_val_acc = 100 * correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        epoch_end_time = time.time() # Time epoch end
        epoch_duration = epoch_end_time - epoch_start_time

        epoch_metrics = { # Metrics for the current epoch
            "type": "metric",
            "epoch": epoch + 1,
            "epochs": epochs,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "duration_sec": epoch_duration
        }
        print(json.dumps(epoch_metrics)) # Print metrics as JSON
        with open(log_file_path, 'a') as f_log: # Append to metrics file
            f_log.write(json.dumps(epoch_metrics) + '\n')
        
        # if scheduler: scheduler.step() # Step the scheduler

    print("Finished Training") # Log end of training
    
    # Save the trained model (optional, can be large)
    # model_save_path = os.path.join(results_dir, f"{run_id}_model.pth")
    # torch.save(model.state_dict(), model_save_path)
    # print(json.dumps({"type": "log", "message": f"Model saved to {model_save_path}"}))

    # Plotting and saving final graphs
    plot_path_acc = os.path.join(results_dir, f"{run_id}_accuracy_plot.png")
    plot_path_loss = os.path.join(results_dir, f"{run_id}_loss_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path_acc)
    plt.close()
    print(json.dumps({"type": "plot", "plot_type": "accuracy", "path": plot_path_acc}))

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path_loss)
    plt.close()
    print(json.dumps({"type": "plot", "plot_type": "loss", "path": plot_path_loss}))

    final_summary = { # Final summary message
        "type": "summary",
        "message": "Training complete. Metrics and plots saved.",
        "final_train_acc": epoch_train_acc,
        "final_val_acc": epoch_val_acc,
        "final_train_loss": epoch_train_loss,
        "final_val_loss": epoch_val_loss,
        "accuracy_plot_path": plot_path_acc,
        "loss_plot_path": plot_path_loss,
        "metrics_log_path": log_file_path
    }
    print(json.dumps(final_summary)) # Print final summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN model on CIFAR-10.") # Argument parser
    parser.add_argument('--model_name', type=str, required=True, choices=['CustomCNN', 'ResNet50', 'VGG16'], help='Name of the model to train.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--augment', type=bool, default=True, help='Whether to use data augmentation.')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results.')
    
    args = parser.parse_args() # Parse arguments

    run_id_str = f"{args.model_name}_{time.strftime('%Y%m%d_%H%M%S')}" # Unique ID for the run

    train_model(args.model_name, args.epochs, args.lr, args.batch_size, args.augment, args.results_dir, run_id_str)
