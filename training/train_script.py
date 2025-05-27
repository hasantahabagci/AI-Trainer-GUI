# Hasan Taha Bağcı
# 150210338
# Model Training Script (to be run by QProcess)

import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add project root to sys.path to allow importing from other project directories
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from utils.data_utils import get_cifar10_loaders # Import data loaders
from models import MODEL_FACTORIES # Import model factories

def train_model(model_name, epochs, batch_size, learning_rate, use_data_augmentation, use_pretrained, device_str): # Main training function
    """
    Trains the specified model on CIFAR-10 with the given parameters.
    Outputs progress and metrics to stdout for GUI parsing.
    """
    print(f"LOG:Starting training for {model_name} on {device_str}...") # Log start
    print(f"LOG:Params: Epochs={epochs}, BatchSize={batch_size}, LR={learning_rate}, Augment={use_data_augmentation}, Pretrained={use_pretrained}")

    # Determine the device
    if device_str == "cuda" and torch.cuda.is_available(): # Check for CUDA
        device = torch.device("cuda")
    elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Check for MPS
        device = torch.device("mps")
        # Fallback for operations not supported on MPS, if necessary
        # Can also check torch.backends.mps.is_built()
    else:
        device = torch.device("cpu") # Default to CPU
        if device_str == "cuda":
            print("LOG:CUDA selected but not available, falling back to CPU.")
        elif device_str == "mps":
            print("LOG:MPS selected but not available/built, falling back to CPU.")
            
    print(f"LOG:Using device: {device}")

    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size, data_augmentation=use_data_augmentation) # Get data loaders

    if model_name not in MODEL_FACTORIES: # Check if model is available
        print(f"ERROR:Model {model_name} not found.")
        return
    
    model = MODEL_FACTORIES[model_name](num_classes=10, pretrained=use_pretrained) # Instantiate model
    model.to(device) # Move model to device

    criterion = nn.CrossEntropyLoss() # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Define optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1) # Learning rate scheduler

    for epoch in range(epochs): # Training loop
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Use tqdm for progress bar, ensuring output is flushed for QProcess
        train_pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs} [Train]", file=sys.stdout, flush=True, leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        for i, data in train_pbar:
            inputs, labels = data[0].to(device), data[1].to(device) # Get inputs and labels
            optimizer.zero_grad() # Zero gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Optimize

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            current_batch_acc = (predicted == labels).sum().item() / labels.size(0) if labels.size(0) > 0 else 0
            train_pbar.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {current_batch_acc:.3f}")
            if i % (len(trainloader) // 10) == 0 or i == len(trainloader) -1 : # Print progress update less frequently for less noise
                 sys.stdout.flush() # Ensure output is sent

        train_loss_epoch = running_loss / len(trainloader) if len(trainloader) > 0 else 0
        train_acc_epoch = 100 * correct_train / total_train if total_train > 0 else 0
        train_pbar.close()

        # Validation
        model.eval() # Set model to evaluation mode
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        val_pbar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Epoch {epoch+1}/{epochs} [Val]", file=sys.stdout, flush=True, leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        with torch.no_grad(): # No gradients needed for validation
            for i, data in val_pbar:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                current_batch_acc_val = (predicted == labels).sum().item() / labels.size(0) if labels.size(0) > 0 else 0
                val_pbar.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {current_batch_acc_val:.3f}")
                if i % (len(testloader) // 10) == 0 or i == len(testloader) - 1:
                    sys.stdout.flush()


        val_loss_epoch = val_loss / len(testloader) if len(testloader) > 0 else 0
        val_acc_epoch = 100 * correct_val / total_val if total_val > 0 else 0
        val_pbar.close()
        
        scheduler.step() # Step the scheduler

        # Output metrics as JSON for GUI parsing
        metrics = {
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "train_loss": train_loss_epoch,
            "train_acc": train_acc_epoch,
            "val_loss": val_loss_epoch,
            "val_acc": val_acc_epoch,
            "lr": optimizer.param_groups[0]['lr']
        }
        print(f"EPOCH_METRIC:{json.dumps(metrics)}")
        sys.stdout.flush() # Important: flush stdout to ensure GUI receives data timely

    print("LOG:Training finished.")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Model Training Script") # Argument parser
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., CustomCNN, ResNet50, VGG16)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and testing")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--use_data_augmentation", action='store_true', help="Enable data augmentation")
    parser.add_argument("--use_pretrained", action='store_true', help="Use pretrained weights for ResNet/VGG")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to use (cpu, cuda, or mps)") # Added mps to choices
    
    args = parser.parse_args()

    train_model(args.model_name, args.epochs, args.batch_size, args.learning_rate, args.use_data_augmentation, args.use_pretrained, args.device)