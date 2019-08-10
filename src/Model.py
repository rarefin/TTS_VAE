import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


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
        self.conv_layers1 = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv_layers2 = nn.Sequential(
            nn.Conv1d(in_channels=in_dim+16, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=1, batch_first=True)

        self.reconstruct = nn.Linear(512, in_dim)

        self.latent2hidden = nn.Linear(16, 512)
        self.tanh = nn.Tanh()

    def forward(self, z, padded_x, sorted_lengths, sorted_idx):

        padded_x = padded_x.view(-1, padded_x.size(-1), padded_x.size(-2))
        convolved_x = self.conv_layers1(padded_x)
        #
        # convolved_x = convolved_x.view(-1, convolved_x.size(-1), convolved_x.size(-2))

        z_new = [a.view(1, -1).repeat(sorted_lengths.int().data.tolist()[i], 1) for i, a in enumerate(z)]
        z_new = torch.nn.utils.rnn.pad_sequence(z_new, batch_first=True)
        convolved_x = torch.cat((convolved_x, z_new), 2)

        convolved_x = self.conv_layers2(convolved_x)
        convolved_x = convolved_x.view(-1, convolved_x.size(-1), convolved_x.size(-2))

        packed_padded_x = pack_padded_sequence(convolved_x, batch_first=True, lengths=sorted_lengths)

        h = self.tanh(self.latent2hidden(z))
        h = h.view(1, z.size(0), 512)
        packed_output, _ = self.lstm(packed_padded_x, [h, h])

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


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownScale(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownScale, self).__init__()
        self.down = nn.Sequential(
            nn.AvgPool1d(2),
            ConvLayer(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UpScale(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UpScale, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = ConvLayer(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff = x2.size(2) - x1.size(2)

        x1 = F.pad(x1, (diff // 2, diff - diff // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(self, in_dim=80, latent_size=16):
        super(UnetDecoder, self).__init__()
        self.conv_layer = ConvLayer(in_dim, in_dim)
        self.inc = ConvLayer(in_dim+latent_size, 192)
        self.down1 = DownScale(192, 384)
        self.down2 = DownScale(384, 768)
        self.down3 = DownScale(768, 768)
        self.up1 = UpScale(1536, 384)
        self.up2 = UpScale(768, 192)
        self.up3 = UpScale(384, 96)
        self.outc = nn.Conv1d(in_channels=96, out_channels=80, kernel_size=3, padding=1)

        # self.reconstruct = nn.Linear(512, in_dim)

    def forward(self, z, padded_x, sorted_lengths, sorted_idx):

        padded_x = padded_x.view(-1, padded_x.size(-1), padded_x.size(-2))
        convolved_x = self.conv_layer(padded_x)

        z_new = [a.view(1, -1).repeat(sorted_lengths.int().data.tolist()[i], 1) for i, a in enumerate(z)]
        z_new = torch.nn.utils.rnn.pad_sequence(z_new, batch_first=True)
        z_new = z_new.view(-1, z_new.size(-1), z_new.size(-2))
        convolved_x = torch.cat((convolved_x, z_new), 1)

        # Downscale through frame dimension
        x1 = self.inc(convolved_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Upscale through frame dimension with skip connection
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        x = x.view(-1, x.size(-1), x.size(-2))

        # project outputs to mel
        # reconstructed_mel = self.reconstruct(padded_outputs.view(-1, padded_outputs.size(2)))
        # reconstructed_mel = reconstructed_mel.view(b, s, 80)

        return x


class VAE(nn.Module):
    def __init__(self, in_channels, latent_size):
        super(VAE, self).__init__()
        self.encode = Encoder(in_channels, latent_size)
        #self.decode = Decoder(in_channels, latent_size)
        self.decode = UnetDecoder(in_channels, latent_size)

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
