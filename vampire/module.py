from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, out_features: int, num_layers: int) -> None:
        super().__init__()
        in_features = vocab_size
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        activations: List[torch.Tensor] = []
        for layer in self.layers:
            x = F.relu(layer(x))
            activations.append(x)
        return x, activations


class MeanProjection(nn.Linear):
    pass


class LogVarianceProjection(nn.Linear):
    pass


class Decoder(nn.Module):
    def __init__(self, in_features: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, vocab_size)

    def forward(self, x):
        return self.linear(x)


class VAE(nn.Module):
    def __init__(
        self, vocab_size: int, out_features: int, encoder_num_layers: int
    ) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, out_features, encoder_num_layers)
        self.mean_projection = MeanProjection(out_features, out_features)
        self.log_variance_projection = LogVarianceProjection(out_features, out_features)
        self.decoder = Decoder(out_features, vocab_size)

    def encode(self, x):
        x = self.encoder(x)
        return self.mean_projection(x), self.log_variance_projection(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return F.softmax(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
