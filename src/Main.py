from DataLoader import MelDataset, collateFunction
from Trainer import train
from Model import VAE
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os

# seed all RNGs for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# CuDNN reproducibility options
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# read the config file
with open("config.json", "r") as read_file:
    config = json.load(read_file)


data_dir = config["paths"]["data_dir"]
file_names = [name for name in os.listdir(data_dir)]


# Split data into train and validation
val_proportion = config["training"]["val_proportion"]
train_file_names, val_file_names = train_test_split(file_names, test_size=val_proportion, random_state=1, shuffle=True)

BATCH_SIZE = config["training"]["batch_size"]
# Create Dataloader
train_dataset = MelDataset(data_dir, train_file_names)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collateFunction)

val_dataset = MelDataset(data_dir, val_file_names)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collateFunction)

# Instantiate model
model = VAE(config["network"]["input_size"], config["network"]["latent_size"])


# Create optimizer and loss
learning_rate = config["training"]["lr"]
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


data_loaders = {'train': train_loader, 'val': val_loader}
model = train(model, data_loaders, optimizer, config)




