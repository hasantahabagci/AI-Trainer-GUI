# Hasan Taha Bağcı 150210338
# run_training.py

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.utils import get_device
from core.data_loader import get_cifar10_loaders
from models.cnn_models import get_model, NUM_CIFAR10_CLASSES
from core.trainer import Trainer # Trainer class

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training Script for CNN Analyzer")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to train (e.g., CustomCNN, ResNet50)")
    parser.add_argument("--pretrained", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use pretrained weights (True/False)")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--optimizer-name", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer to use")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (e.g., cpu, cuda, mps)")
    parser.add_argument("--augment-data", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to augment training data")

    args = parser.parse_args()

    print(f"run_training.py started with args: {args}") # Log arguments

    selected_device = get_device(args.device) # Get the actual torch.device object
    print(f"Using device: {selected_device}")

    print("Loading CIFAR-10 data...")
    try:
        train_loader, val_loader, classes = get_cifar10_loaders(
            batch_size=args.batch_size,
            augment=args.augment_data,
            num_workers=2, 
            pin_memory=True if selected_device.type != 'cpu' else False
        )
        print(f"Data loaded. Classes: {classes}. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)


    print(f"Initializing model: {args.model_name} (Pretrained: {args.pretrained})")
    try:
        # CustomCNN does not take 'pretrained' in its constructor directly via get_model's current setup
        if args.model_name == "CustomCNN":
             model = get_model(args.model_name, num_classes=NUM_CIFAR10_CLASSES)
        else:
             model = get_model(args.model_name, num_classes=NUM_CIFAR10_CLASSES, pretrained=args.pretrained)
        print("Model initialized.")
    except Exception as e:
        print(f"Error initializing model: {e}", file=sys.stderr)
        sys.exit(1)

    # Trainer Config
    trainer_config = {
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "optimizer_name": args.optimizer_name,
        "criterion_name": "CrossEntropyLoss" 
    }

    # Initialize Trainer
    print("Initializing Trainer...")
    try:
        trainer = Trainer(model, train_loader, val_loader, selected_device, trainer_config)
        print("Trainer initialized.")
    except Exception as e:
        print(f"Error initializing trainer: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("Starting training via Trainer...")
    try:
        trainer.train()
        print("Training script finished successfully.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("Training interrupted by user (KeyboardInterrupt).", file=sys.stderr)

        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) # Print full traceback to stderr
        sys.exit(1)

if __name__ == '__main__':
    main()
