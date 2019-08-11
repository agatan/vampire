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
        self.vocab_size = vocab_size
        self.linear = nn.Linear(in_features, vocab_size, bias=False)

    def forward(self, x):
        return self.linear(x)

    def extract_topics(self, top_k: int = 20) -> List[Tuple[str, List[int]]]:
        """extract_topics returns a list of (topic name, word indices)
        """
        words = list(range(self.vocab_size))
        topics: List[Tuple[str, List[int]]] = []
        for i, topic in enumerate(self.linear.weight.detach()):  # type: ignore
            word_strengths = list(zip(words, topic.tolist()))
            sorted_by_strength = sorted(
                word_strengths, key=lambda x: x[1], reverse=True
            )
            top_k_indices = [x[0] for x in sorted_by_strength][:top_k]
            topics.append((str(i), top_k_indices))
        return topics


class VAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        out_features: int,
        encoder_num_layers: int,
        z_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, out_features, encoder_num_layers)
        self.mean_projection = MeanProjection(out_features, out_features)
        self.log_variance_projection = LogVarianceProjection(out_features, out_features)
        self.decoder = Decoder(out_features, vocab_size)
        self.z_dropout = nn.Dropout(z_dropout)

    def encode(self, x):
        x, _ = self.encoder(x)
        return self.mean_projection(x), self.log_variance_projection(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        theta = self.z_dropout(z)
        return torch.softmax(theta, dim=-1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
