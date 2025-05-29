# Hasan Taha Bağcı 150210338
# core/utils.py

import torch

def get_available_devices(): # Returns a list of available PyTorch devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            devices.append('mps')
    except AttributeError:
        pass # 'mps' won't be added if not available or detectable
    return devices

def get_device(preferred_device=None): # Selects the best available device or a preferred one
    available_devices = get_available_devices()
    
    if preferred_device and preferred_device in available_devices:
        print(f"Using preferred device: {preferred_device}")
        return torch.device(preferred_device)

    if 'cuda' in available_devices:
        print("Using device: cuda")
        return torch.device('cuda')
    elif 'mps' in available_devices:
        print("Using device: mps")
        return torch.device('mps')
    else:
        print("Using device: cpu")
        return torch.device('cpu')

if __name__ == '__main__':
    # Test the functions
    print("Available PyTorch devices:", get_available_devices())
    selected_device = get_device()
    print(f"Selected device for training: {selected_device}")
    
    # Test with preference
    selected_device_mps = get_device('mps')
    print(f"Selected device with preference 'mps': {selected_device_mps}")

    selected_device_cuda = get_device('cuda')
    print(f"Selected device with preference 'cuda': {selected_device_cuda}")

    selected_device_cpu = get_device('cpu')
    print(f"Selected device with preference 'cpu': {selected_device_cpu}")
