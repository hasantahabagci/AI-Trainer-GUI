# Hasan Taha Bağcı
# 150210338
# Models Package Initializer

from .custom_cnn_model import get_custom_cnn
from .resnet_model import get_resnet50
from .vgg_model import get_vgg16

MODEL_FACTORIES = { # Dictionary of model factory functions
    "CustomCNN": get_custom_cnn,
    "ResNet50": get_resnet50,
    "VGG16": get_vgg16,
}