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

def train_model(model_name, epochs, lr, batch_size, augment, results_dir, run_id): # Ana eğitim fonksiyonu
    """
    Belirtilen CNN modelini CIFAR-10 üzerinde eğitir.
    Metrikleri loglar ve grafikleri kaydeder, batch seviyesinde ilerleme dahil.
    """
    # Erken log mesajları GUI'ye bilgi vermek için
    print(json.dumps({"type": "log", "message": f"Training process {run_id} started."}), flush=True)
    print(json.dumps({"type": "log", "message": "Configuration:"}), flush=True)
    print(json.dumps({"type": "log", "message": f"  Model: {model_name}"}), flush=True)
    print(json.dumps({"type": "log", "message": f"  Epochs: {epochs}"}), flush=True)
    print(json.dumps({"type": "log", "message": f"  Learning Rate: {lr}"}), flush=True)
    print(json.dumps({"type": "log", "message": f"  Batch Size: {batch_size}"}), flush=True)
    print(json.dumps({"type": "log", "message": f"  Data Augmentation: {augment}"}), flush=True)
    print(json.dumps({"type": "log", "message": f"  Results Directory: {results_dir}"}), flush=True)
    
    os.makedirs(results_dir, exist_ok=True) # Sonuçlar dizinini oluştur
    log_file_path = os.path.join(results_dir, f"{run_id}_metrics.jsonl") # Metrik log dosyası yolu

    print(json.dumps({"type": "log", "message": "Setting up device..."}), flush=True)
    if torch.backends.mps.is_available(): # Apple Silicon MPS kontrolü
        device = torch.device("mps")
    elif torch.cuda.is_available(): # CUDA kontrolü
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") # CPU'ya varsayılan
    print(json.dumps({"type": "log", "message": f"Using device: {device}"}), flush=True) # Kullanılan cihazı logla

    print(json.dumps({"type": "log", "message": "Loading CIFAR-10 dataset..."}), flush=True) # Veri yükleme logu
    trainloader, testloader, classes = get_cifar10_loaders(batch_size=batch_size, augment=augment)
    if trainloader is None or testloader is None:
        error_msg = "Failed to load dataset. Exiting."
        print(json.dumps({"type": "error", "message": error_msg}), flush=True) # Hatayı JSON olarak gönder
        return
    print(json.dumps({"type": "log", "message": "CIFAR-10 dataset loaded."}), flush=True)

    print(json.dumps({"type": "log", "message": f"Initializing model: {model_name}..."}), flush=True) # Model başlatma logu
    if model_name == 'CustomCNN':
        model = CustomCNN(num_classes=len(classes))
    elif model_name == 'ResNet50':
        model = get_resnet50(num_classes=len(classes), pretrained=True) 
    elif model_name == 'VGG16':
        model = get_vgg16(num_classes=len(classes), pretrained=True) 
    else:
        error_msg = f"Unknown model: {model_name}"
        print(json.dumps({"type": "error", "message": error_msg}), flush=True) # Hatayı JSON olarak gönder
        return
    model.to(device) # Modeli cihaza taşı
    print(json.dumps({"type": "log", "message": f"Model {model_name} initialized and moved to {device}."}), flush=True)

    criterion = nn.CrossEntropyLoss() # Kayıp fonksiyonunu tanımla
    optimizer = optim.Adam(model.parameters(), lr=lr) # Optimizatörü tanımla

    history = { # Eğitim geçmişini saklamak için sözlük
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print(json.dumps({"type": "log", "message": "Starting training loop..."}), flush=True) # Eğitim başlangıcı logu
    total_batches_in_epoch = len(trainloader) # Epoch başına toplam batch sayısı

    for epoch in range(epochs): # Veri kümesi üzerinde birden çok kez döngü
        model.train() # Modeli eğitim moduna ayarla
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        epoch_start_time = time.time() # Epoch başlangıç zamanı

        # Epoch başladığında GUI'ye bir batch_progress mesajı göndererek progress bar'ı sıfırlamasını sağlayabiliriz.
        # Bu, update_epoch_status_display'deki mantıkla birleştiğinde iyi çalışmalı.
        initial_batch_progress_for_new_epoch = {
            "type": "batch_progress",
            "epoch": epoch + 1,
            "epochs": epochs,
            "current_batch": 0, # Yeni epoch için 0. batch
            "total_batches": total_batches_in_epoch,
            "batch_loss": 0.0 # Başlangıçta kayıp yok
        }
        print(json.dumps(initial_batch_progress_for_new_epoch), flush=True)


        for i, data in enumerate(trainloader, 0): # Eğitim verileri üzerinde döngü
            current_batch_num = i + 1
            inputs, labels = data[0].to(device), data[1].to(device) # Girdileri ve etiketleri cihaza al

            optimizer.zero_grad() # Parametre gradyanlarını sıfırla

            outputs = model(inputs) # İleri geçiş
            loss = criterion(outputs, labels) # Kaybı hesapla
            loss.backward() # Geri geçiş
            optimizer.step() # Optimize et

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Batch ilerleme güncellemesi gönder
            batch_progress_data = {
                "type": "batch_progress",
                "epoch": epoch + 1,
                "epochs": epochs,
                "current_batch": current_batch_num,
                "total_batches": total_batches_in_epoch,
                "batch_loss": loss.item()
            }
            print(json.dumps(batch_progress_data), flush=True) # Batch ilerlemesini JSON olarak yazdır (flush=True ile)

        epoch_train_loss = running_loss / total_batches_in_epoch 
        epoch_train_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        model.eval() # Modeli değerlendirme moduna ayarla
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Doğrulama için gradyan gerekmez
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss_val_batch = criterion(outputs, labels) 
                val_loss += loss_val_batch.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(testloader)
        epoch_val_acc = 100 * correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        epoch_end_time = time.time() # Epoch bitiş zamanı
        epoch_duration = epoch_end_time - epoch_start_time

        epoch_metrics = { # Mevcut epoch için metrikler
            "type": "metric",
            "epoch": epoch + 1,
            "epochs": epochs,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "duration_sec": epoch_duration
        }
        print(json.dumps(epoch_metrics), flush=True) # Metrikleri JSON olarak yazdır (flush=True ile)
        with open(log_file_path, 'a') as f_log: # Metrik dosyasına ekle
            f_log.write(json.dumps(epoch_metrics) + '\n')
        
    print(json.dumps({"type": "log", "message": "Finished Training Loop."}), flush=True) # Eğitim sonu logu
    
    plot_path_acc = os.path.join(results_dir, f"{run_id}_accuracy_plot.png")
    plot_path_loss = os.path.join(results_dir, f"{run_id}_loss_plot.png")

    try:
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
        print(json.dumps({"type": "plot", "plot_type": "accuracy", "path": plot_path_acc}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"Failed to save accuracy plot: {str(e)}"}), flush=True)


    try:
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
        print(json.dumps({"type": "plot", "plot_type": "loss", "path": plot_path_loss}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"Failed to save loss plot: {str(e)}"}), flush=True)


    final_summary = { # Nihai özet mesajı
        "type": "summary",
        "message": "Training complete. Metrics and plots saved.",
        "final_train_acc": epoch_train_acc if 'epoch_train_acc' in locals() else "N/A", # Handle cases where training might not complete an epoch
        "final_val_acc": epoch_val_acc if 'epoch_val_acc' in locals() else "N/A",
        "final_train_loss": epoch_train_loss if 'epoch_train_loss' in locals() else "N/A",
        "final_val_loss": epoch_val_loss if 'epoch_val_loss' in locals() else "N/A",
        "accuracy_plot_path": plot_path_acc,
        "loss_plot_path": plot_path_loss,
        "metrics_log_path": log_file_path
    }
    print(json.dumps(final_summary), flush=True) # Nihai özeti yazdır (flush=True ile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN model on CIFAR-10.") # Argüman ayrıştırıcısı
    parser.add_argument('--model_name', type=str, required=True, choices=['CustomCNN', 'ResNet50', 'VGG16'], help='Name of the model to train.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    # Argparse boolean argümanları için daha iyi bir yol: store_true/store_false
    parser.add_argument('--augment', dest='augment', action='store_true', help='Enable data augmentation (default).')
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='Disable data augmentation.')
    parser.set_defaults(augment=True) # Varsayılan olarak augment True
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results.')
    
    args = parser.parse_args() # Argümanları ayrıştır

    run_id_str = f"{args.model_name}_{time.strftime('%Y%m%d_%H%M%S')}" # Çalıştırma için benzersiz ID

    train_model(args.model_name, args.epochs, args.lr, args.batch_size, args.augment, args.results_dir, run_id_str)
