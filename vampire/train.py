import os

import pytorch_lightning as plt
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import HyperOptArgumentParser, Experiment

from .data import BoWDataset, LivedoorNewsDataset, train_test_split
from .preprocessing import MeCabTokenizer, Vocab


class Vampire(plt.LightningModule):
    def __init__(self, hparams) -> None:
        dataset = LivedoorNewsDataset(hparams.data_root)
        dataset, _ = train_test_split(dataset)
        self.train_dataset, self.val_dataset = train_test_split(dataset)
        self.tokenizer = MeCabTokenizer()
        tokens = (self.tokenizer.tokenize(s) for s, _ in self.train_dataset)
        self.vocab = Vocab(tokens, max_size=hparams.max_vocab)

    @staticmethod
    def add_model_specific_args(parent: HyperOptArgumentParser, root_dir):
        parser = HyperOptArgumentParser(strategy=parent.strategy, parents=[parent])
        parser.add_argument(
            "--data_root",
            default=os.path.join(root_dir, "livedoor-news-corpus", "text"),
        )
        parser.add_argument("--max_vocab", default=30000, type=int)
        return parser

    @plt.data_loader
    def tng_dataloader(self) -> BoWDataset:
        return BoWDataset(self.tokenizer, self.vocab, self.train_dataset)

    @plt.data_loader
    def val_dataloader(self) -> BoWDataset:
        return BoWDataset(self.tokenizer, self.vocab, self.val_dataset)


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
    parser = Vampire.add_model_specific_args(parent_parser, ".")
    hparams = parser.parse_args()

    model = Vampire(hparams)

    exp = Experiment(
        name=hparams.experiment_name,
        save_dir=hparams.test_tube_save_path,
        autosave=False,
    )
    exp.argparse(hparams)
    exp.save()

    trainer = Trainer(experiment=exp)
    trainer.fit(model)


if __name__ == "__main__":
    main()
