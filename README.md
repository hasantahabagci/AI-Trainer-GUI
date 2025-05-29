# AI-Trainer-GUI

**Author:** Hasan Taha **Bağcı (150210338)**

## Project Summary

![alt text](https://github.com/hasantahabagci/AI-Trainer-GUI/content/sample.png?raw=true "AI-Trainer-GUI Screenshot")

AI-Trainer-GUI is a Python application designed for the comparative analysis of Convolutional Neural Network (CNN) architectures for image classification. This project focuses on evaluating the performance of ResNet50, VGG16, and a custom-built CNN on the CIFAR-10 dataset.

The core of the project is a user-friendly Graphical User Interface (GUI) built with PyQt5. This interface allows users to:

* Select a CNN model (ResNet50, VGG16, or CustomCNN).
* Configure training parameters such as epochs, learning rate, batch size, and optimizer.
* Choose to use pretrained weights for applicable models (ResNet50, VGG16).
* Select the computation device (CPU, MPS for Apple Silicon, CUDA if available).
* Enable or disable data augmentation for the training set.
* Start and stop the training process.
* View real-time training logs.
* Observe dynamic plots of training and validation accuracy and loss as the model trains.

The application aims to provide an accessible environment for non-programmers to experiment with different CNN models, observe their training dynamics, and compare their performance metrics, thereby facilitating a better understanding of CNN behavior.

## Project Structure

The project is organized into the following main directories:

* `core/`: Contains the core logic for data loading (`data_loader.py`), device utilities (`utils.py`), and the training engine (`trainer.py`).
* `models/`: Defines the CNN architectures (`cnn_models.py`), including the custom CNN, ResNet50, and VGG16.
* `gui/`: Includes the PyQt5 components for the main application window (`main_window.py`) and the real-time plotting widget (`plot_widget.py`).
* Root Directory: Contains the main entry script (`main.py`), the training execution script (`run_training.py`), and the requirements file (`requirements.txt`).

## Requirements

The project requires Python 3.11.11 and the following packages:

* `torch`
* `torchvision`
* `torchaudio`
* `PyQt5`
* `matplotlib`
* `numpy`
* `tqdm`

You can install these dependencies using the provided `requirements.txt` file.

## Setup and Installation

1. **Clone the Repository (or Download Files):**
   Ensure all project files are downloaded and organized according to the project structure described above.

2. **Navigate to Project Directory:**
   Open your terminal or command prompt and change to the root directory of the project (e.g., `AI-Trainer-GUI/` depending on how you've named it).

   ```
   cd path/to/your/project/AI-Trainer-GUI
   ```

3. **Create and Activate a Virtual Environment (Recommended):**
   It's good to use a virtual environment to manage project dependencies.

   ```
   # Create a virtual environment (e.g., named 'venv')
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   # venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```

4. **Install Dependencies:**
   With your virtual environment activated (if you created one), install the required packages:

   ```
   pip install -r requirements.txt
   ```

## How to Run the Application

Once the setup is complete, you can run the application using the main script:

```
python main.py
```

This will launch the AI-Trainer-GUI. You can then:

1. Select the desired CNN model from the "Select Model" dropdown.
2. Adjust training parameters (epochs, learning rate, etc.).
3. Choose the computation device (CPU, MPS, or CUDA if available and detected).
4. Click the "Start Training" button.

The training logs and performance plots (loss and accuracy for training and validation sets) will update in real-time.
