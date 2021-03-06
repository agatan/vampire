from typing import Tuple, Optional, Union, List
from pathlib import Path
import abc

import torch
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection as model_selection

from .preprocessing import Tokenizer, Vocab


class Dataset(abc.ABC, data.Dataset):
    @abc.abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, Optional[int]]:
        pass


class LivedoorNewsDataset(Dataset):
    def __init__(self, basedir: Union[str, Path]) -> None:
        self.basedir = Path(basedir)
        labels: List[str] = []
        paths: List[Path] = []
        for child in self.basedir.iterdir():
            if not child.is_dir():
                continue
            label = child.stem
            for filepath in child.iterdir():
                labels.append(label)
                paths.append(filepath)
        self.paths = paths
        self.label_encoder = LabelEncoder().fit(labels)
        self.labels: List[int] = self.label_encoder.transform(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[str, Optional[int]]:
        label = self.labels[index]
        path = self.paths[index]
        content = path.read_text(encoding="utf-8")
        content = self._trim_header(content)
        return content, label

    def _trim_header(self, content: str) -> str:
        lines = content.split("\n", 3)
        return lines[-1]


class BoWDataset(data.Dataset):
    def __init__(self, tokenizer: Tokenizer, vocab: Vocab, dataset: Dataset) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        s, label = self.dataset[index]
        bow = self.vocab.encode_bow(self.tokenizer.tokenize(s))
        bow_tensor = torch.tensor(bow).float()
        if label is None:
            return bow_tensor, None
        else:
            return bow_tensor, torch.tensor(label)


def train_test_split(x, test_size=0.1, shuffle=True, random_state=42):
    ys = [y for _, y in x]
    return model_selection.train_test_split(
        x, test_size=test_size, shuffle=shuffle, stratify=ys, random_state=random_state
    )
