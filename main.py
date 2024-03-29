import torch

import train
from model import CursedNet


# Declare Global Settings
root_dir = "dataset"
learning_rate = 0.01
epochs = 500
percentages = [0.85, 0.15, 0.00] # Training, Validation, Testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Declare Interation Settings
interations = [
    {"model": CursedNet, "batch_size": 2, "save_dir": "runs/PROTO_V2_IMAG"},
]

if __name__ == "__main__":
    for data in interations:
        train.run(data['model'], root_dir, data['save_dir'], data['batch_size'],
                  learning_rate, epochs, percentages, device)