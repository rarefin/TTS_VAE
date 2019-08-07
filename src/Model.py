import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_size):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)

        self.mu = nn.Linear(512, latent_size)
        self.log_var = nn.Linear(512, latent_size)

    def forward(self, padded_x, sorted_lengths):
        padded_x = padded_x.view(-1, padded_x.size(-1), padded_x.size(-2))
        convolved_x = self.conv_layers(padded_x)

        convolved_x = convolved_x.view(-1, convolved_x.size(-1), convolved_x.size(-2))

        packed_padded_x = pack_padded_sequence(convolved_x, batch_first=True, lengths=sorted_lengths)

        packed_output, _ = self.lstm(packed_padded_x)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # global average pooling
        # avg_pool = torch.mean(output, 1)
        avg_pool = torch.sum(output, 1) / sorted_lengths.repeat(512, 1).transpose(1, 0)

        mu = self.mu(avg_pool)
        log_var = self.log_var(avg_pool)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, in_dim=80, latent_size=16):
        super(Decoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128+latent_size, hidden_size=512, num_layers=2, batch_first=True)

        self.reconstruct = nn.Linear(512, in_dim)

    def forward(self, z, padded_x, sorted_lengths, sorted_idx):

        padded_x = padded_x.view(-1, padded_x.size(-1), padded_x.size(-2))
        convolved_x = self.conv_layers(padded_x)

        convolved_x = convolved_x.view(-1, convolved_x.size(-1), convolved_x.size(-2))

        z = [a.view(1, -1).repeat(sorted_lengths.int().data.tolist()[i], 1) for i, a in enumerate(z)]
        z = torch.nn.utils.rnn.pad_sequence(z, batch_first=True)
        convolved_x = torch.cat((convolved_x, z), 2)

        packed_padded_x = pack_padded_sequence(convolved_x, batch_first=True, lengths=sorted_lengths)

        packed_output, _ = self.lstm(packed_padded_x)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # process outputs
        padded_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # project outputs to mel
        reconstructed_mel = self.reconstruct(padded_outputs.view(-1, padded_outputs.size(2)))
        reconstructed_mel = reconstructed_mel.view(b, s, 80)

        return reconstructed_mel


class VAE(nn.Module):
    def __init__(self, in_channels, latent_size):
        super(VAE, self).__init__()
        self.encode = Encoder(in_channels, latent_size)
        self.decode = Decoder(in_channels, latent_size)

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(mu)

        return eps * std + mu

    def forward(self, padded_mel, mel_seq_lengths, padded_text_features):
        sorted_lengths, sorted_idx = torch.sort(mel_seq_lengths, descending=True)
        padded_mel = padded_mel[sorted_idx]

        mu, log_var = self.encode(padded_mel, sorted_lengths)

        z = self.reparameterize(mu, log_var)

        reconstructed_mel = self.decode(z, padded_text_features, sorted_lengths, sorted_idx)

        return reconstructed_mel, mu, log_var, z
