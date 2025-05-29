# Hasan Taha Bağcı 150210338
# core/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import json 

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation/test set.
            device (torch.device): The device to train on (e.g., 'cuda', 'mps', 'cpu').
            config (dict): Configuration dictionary containing:
                - learning_rate (float)
                - num_epochs (int)
                - optimizer_name (str, e.g., "Adam", "SGD")
                - criterion_name (str, e.g., "CrossEntropyLoss")
                # Add other optimizer/criterion specific params if needed
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Initialize criterion
        if config.get("criterion_name", "CrossEntropyLoss") == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion: {config.get('criterion_name')}")
        
        # Initialize optimizer
        optimizer_name = config.get("optimizer_name", "Adam")
        lr = config.get("learning_rate", 0.001)

        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9) # Common momentum for SGD
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.num_epochs = config.get("num_epochs", 10)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self._stop_training_flag = False # Flag to allow stopping training externally

    def _train_one_epoch(self, epoch_num): # Train the model for one epoch
        self.model.train() 
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Using tqdm for progress bar
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num+1}/{self.num_epochs} [Training]", leave=False)
        
        for inputs, labels in progress_bar:
            if self._stop_training_flag: 
                print("Training stopped prematurely by user.")
                progress_bar.close()
                return None, None 

            inputs, labels = inputs.to(self.device), labels.to(self.device) # Move data to the selected device
            
            self.optimizer.zero_grad() 
            
            outputs = self.model(inputs) # Forward pass
            loss = self.criterion(outputs, labels) # Calculate loss
            
            loss.backward() # Backward pass
            self.optimizer.step() 
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct_predictions / total_samples if total_samples > 0 else 0)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
        return epoch_loss, epoch_acc

    def _validate_one_epoch(self, epoch_num):
        self.model.eval() # Set the model to evaluation mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch_num+1}/{self.num_epochs} [Validation]", leave=False)
        
        with torch.no_grad(): # No need to track gradients during validation
            for inputs, labels in progress_bar:
                if self._stop_training_flag:
                    print("Validation stopped prematurely by user.")
                    progress_bar.close()
                    return None, None 

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                progress_bar.set_postfix(loss=loss.item(), acc=100. * correct_predictions / total_samples if total_samples > 0 else 0)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
        return epoch_loss, epoch_acc

    def train(self): # Main training loop
        print(f"Starting training on device: {self.device}")
        self._stop_training_flag = False # Reset stop flag at the beginning of training

        for epoch in range(self.num_epochs):
            if self._stop_training_flag:
                print(f"Training interrupted before epoch {epoch+1}.")
                break

            train_loss, train_acc = self._train_one_epoch(epoch)
            if train_loss is None: # Training was stopped
                break 
            
            val_loss, val_acc = self._validate_one_epoch(epoch)
            if val_loss is None: # Validation was stopped (implies training was stopped)
                break

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            metrics_data = {
                "epoch": epoch + 1,
                "total_epochs": self.num_epochs,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            print(f"METRICS_UPDATE:{json.dumps(metrics_data)}") 

            print(f"Epoch {epoch+1}/{self.num_epochs} Summary: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if not self._stop_training_flag:
            print("Training completed.")
        else:
            print("Training finished due to stop signal.")
            
        return self.history

    def stop_training(self):
        print("Stop signal received by trainer.")
        self._stop_training_flag = True


if __name__ == '__main__':
    print("Trainer class defined. To test, run from a main script that sets up: ")
    print("1. A model (e.g., CustomCNN from models.cnn_models)")
    print("2. CIFAR-10 data loaders (from core.data_loader)")
    print("3. Device (from core.utils)")
    print("4. Configuration dictionary (epochs, lr, optimizer, criterion)")
    
    """
    from core.utils import get_device
    from core.data_loader import get_cifar10_loaders
    from models.cnn_models import get_model # Assuming get_model can fetch CustomCNN

    if __name__ == '__main__':
        print("Setting up dummy trainer test...")
        device = get_device() # Get the best available device
        
        # Dummy config
        config = {
            "learning_rate": 0.001,
            "num_epochs": 2, # Keep epochs low for a quick test
            "optimizer_name": "Adam",
            "criterion_name": "CrossEntropyLoss",
            "model_name": "CustomCNN", # Or ResNet50, VGG16
            "batch_size": 32, # Smaller batch for faster test
            "augment_data": True
        }

        print(f"Using device: {device}")

        # Load data
        print("Loading CIFAR-10 data...")
        train_loader, val_loader, _ = get_cifar10_loaders(
            batch_size=config["batch_size"], 
            augment=config["augment_data"]
        )
        print("Data loaded.")

        # Initialize model
        print(f"Initializing model: {config['model_name']}")
        # For torchvision models, pretrained=False for a quick scratch test
        model = get_model(config['model_name'], num_classes=10, pretrained=False if config['model_name'] != "CustomCNN" else None)
        print("Model initialized.")

        # Initialize Trainer
        trainer = Trainer(model, train_loader, val_loader, device, config)
        print("Trainer initialized. Starting training...")
        
        history = trainer.train()
        
        print("\nTraining finished. History:")
        for key, values in history.items():
            print(f"{key}: {values}")
        
        # Example of how to test stop_training (would require threading or a more complex setup)
        # import threading
        # import time
        # def delayed_stop(trainer_instance, delay):
        # time.sleep(delay)
        # print(f"Calling stop_training() after {delay} seconds from another thread.")
        # trainer_instance.stop_training()
        #
        # stop_thread = threading.Thread(target=delayed_stop, args=(trainer, 5)) # Stop after 5s
        # stop_thread.start()
        # history_interrupted = trainer.train() # This will run until stopped
        # stop_thread.join()
        # print("\nInterrupted training finished. History:")
        # for key, values in history_interrupted.items():
        # print(f"{key}: {values}")
    """
    print("\nTo fully test, uncomment and adapt the pseudo-code above, ensuring all imports are correct.")

