from torch.utils.data import Dataset
import numpy as np
import os
import torch


class MelDataset(Dataset):
    def __init__(self, directory, file_names):
        super(MelDataset, self).__init__()
        self.directory = directory
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        mel = np.load(os.path.join(self.directory, self.file_names[index]))

        return torch.from_numpy(mel)


def collateFunction(batch):
    '''
    Padds batch of variable length

    '''
    ## get sequence lengths
    lengths = torch.Tensor([s.size(0) for s in batch])
    ## padd
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    ## compute mask
    mask = (batch != 0)
    return batch, lengths, mask


