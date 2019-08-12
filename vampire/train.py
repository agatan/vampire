import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import pytorch_lightning as plt
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import HyperOptArgumentParser, Experiment

from .data import BoWDataset, LivedoorNewsDataset, train_test_split
from .preprocessing import MeCabTokenizer, Vocab
from .module import VAE


class VAMPIRE(plt.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        dataset = LivedoorNewsDataset(hparams.data_root)
        dataset, _ = train_test_split(dataset)
        self.train_dataset, self.val_dataset = train_test_split(dataset)
        self.tokenizer = MeCabTokenizer()
        tokens = (self.tokenizer.tokenize(s) for s, _ in self.train_dataset)
        self.vocab = Vocab(tokens, max_size=hparams.max_vocab)
        self.vae = VAE(
            self.vocab.vocab_size, hparams.latent_dim, hparams.encoder_num_layers
        )
        self._kld_linear_scaling = 100.0
        self._kld_weight = 1.0 / self._kld_linear_scaling

        self.background_log_frequency = nn.Parameter(
            self.vocab.background_log_frequency(), requires_grad=False
        )
        self.bow_bn = torch.nn.BatchNorm1d(
            self.vocab.vocab_size, eps=0.001, momentum=0.001, affine=True
        )
        self.bow_bn.weight.data.copy_(
            torch.ones(self.vocab.vocab_size, dtype=torch.float)
        )
        self.bow_bn.weight.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent: HyperOptArgumentParser, root_dir):
        parser = HyperOptArgumentParser(strategy=parent.strategy, parents=[parent])
        parser.add_argument(
            "--data_root",
            default=os.path.join(root_dir, "livedoor-news-corpus", "text"),
        )
        parser.add_argument("--max_vocab", default=30000, type=int)
        parser.add_argument("--latent_dim", default=80, type=int)
        parser.add_argument("--encoder_num_layers", default=2, type=int)
        return parser

    def on_epoch_end(self):
        self._kld_weight = min(1, self.current_epoch / self._kld_linear_scaling)

    def reconstruct_loss(self, x, recon_x):
        recon_x = self.bow_bn(recon_x + self.background_log_frequency)
        recon_x = recon_x + self.background_log_frequency
        log_recon_x = F.log_softmax(recon_x + 1e-10, dim=-1)
        return -torch.sum(x * log_recon_x, dim=-1)

    def kl_divergence_loss(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kld

    def compute_loss(self, x, recon_x, mu, logvar):
        recon_loss = self.reconstruct_loss(x, recon_x)
        kld_loss = self.kl_divergence_loss(mu, logvar)
        loss = self._kld_weight * kld_loss + recon_loss
        return recon_loss.mean(), kld_loss.mean(), loss.mean()

    def training_step(self, data_batch, _batch_nb):
        x, _ = data_batch
        recon_x, mu, logvar = self.vae(x)
        recon_loss, kld_loss, loss = self.compute_loss(x, recon_x, mu, logvar)
        return {
            "loss": loss,
            "prog": {"tng_loss": loss, "recon_loss": recon_loss, "kld_loss": kld_loss},
        }

    def validation_step(self, data_batch, _batch_nb):
        x, _ = data_batch
        recon_x, mu, logvar = self.vae(x)
        recon_loss, kld_loss, loss = self.compute_loss(x, recon_x, mu, logvar)
        return {
            "val_loss": loss,
            "val_recon_loss": recon_loss,
            "val_kld_loss": kld_loss,
        }

    def validation_end(self, outputs):
        values: Dict[str, torch.Tensor] = {}
        for output in outputs:
            for k, v in output.items():
                if k not in values:
                    values[k] = v
                else:
                    values[k] += v
        return {k: v.item() / len(outputs) for k, v in values.items()}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters())]

    @plt.data_loader
    def tng_dataloader(self) -> BoWDataset:
        return torch.utils.data.DataLoader(
            BoWDataset(self.tokenizer, self.vocab, self.train_dataset),
            batch_size=64,
            shuffle=True,
        )

    @plt.data_loader
    def val_dataloader(self) -> BoWDataset:
        return torch.utils.data.DataLoader(
            BoWDataset(self.tokenizer, self.vocab, self.val_dataset),
            batch_size=64,
            shuffle=True,
        )

    @plt.data_loader
    def test_dataloader(self) -> BoWDataset:
        return torch.utils.data.DataLoader(
            BoWDataset(self.tokenizer, self.vocab, self.val_dataset),
            batch_size=64,
            shuffle=True,
        )


def main():
    parent_parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)
    logdir = "logs"
    parent_parser.add_argument(
        "--test_tube_save_path", default=os.path.join(logdir, "test_tube_data")
    )
    parent_parser.add_argument(
        "--model_save_path", default=os.path.join(logdir, "model_weights")
    )
    parent_parser.add_argument(
        "--experiment_name", default=os.path.join(logdir, "vampire")
    )
    parser = VAMPIRE.add_model_specific_args(parent_parser, ".")
    hparams = parser.parse_args()

    model = VAMPIRE(hparams)

    exp = Experiment(
        name=hparams.experiment_name,
        save_dir=hparams.test_tube_save_path,
        autosave=False,
    )
    exp.argparse(hparams)
    exp.save()

    trainer = Trainer(experiment=exp, fast_dev_run=False)
    trainer.fit(model)


if __name__ == "__main__":
    main()
