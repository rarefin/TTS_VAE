import time
import torch
import copy
import os
import numpy as np
from torch.optim import lr_scheduler
import torch.nn as nn


def train(model, dataloaders, optimizer, config):
    epoch_start = config["training"]["epoch_start"]
    num_epochs = config["training"]["num_epochs"]
    last_checkpoint_path = config["training"]["last_checkpoint_path"]

    log_file_path = config["paths"]["log_file_path"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if epoch_start > 0:
        pretrained_dict = torch.load(last_checkpoint_path)
        model.load_state_dict(pretrained_dict)

    criterion1 = nn.MSELoss()
    # Select the device: gpu or cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    criterion1.to(device)

    log_file = open(log_file_path, "a")
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=5)
    start = time.time()
    for epoch in range(epoch_start, epoch_start + num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        log_file.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        log_file.write('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_rec_loss = 0.0
            running_kl_loss = 0.0

            # Iterate over data.
            for X_batch, lengths, mask in dataloaders[phase]:
                X_batch = X_batch.to(device)
                lengths = lengths.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    reconstructed_mel, mu, log_var, z = model(X_batch, lengths, X_batch)
                    rec_loss = criterion1(reconstructed_mel, X_batch)
                    # We will scale the following losses with this factor
                    # scaling_factor = out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3]

                    ####KL divergence loss
                    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))
                    # kl_loss /= scaling_factor

                    loss = rec_loss + kl_loss

                if phase == 'train':
                    loss.backward()
                    #                         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                running_kl_loss += kl_loss.item() * X_batch.size(0)
                running_rec_loss += rec_loss.item() * X_batch.size(0)

            loss = running_loss / len(dataloaders[phase].dataset)
            kl_loss = running_kl_loss / len(dataloaders[phase].dataset)
            rec_loss = running_rec_loss / len(dataloaders[phase].dataset)

            line = '{} Total Loss: {:.6f}, KL Loss: {:.6}, Reconstruction Loss: {:.6}'.format(phase, loss, kl_loss, rec_loss)
            print(line)
            log_file.write(line + '\n')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir + '/epoch_{}.pth'.format(epoch))
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    log_file.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    log_file.close()

    return model
