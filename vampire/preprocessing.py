from typing import List, Set, Iterable, Counter, Optional
import abc

import MeCab
import torch


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, s: str) -> List[str]:
        pass


class MeCabTokenizer(Tokenizer):
    def __init__(
        self,
        part_of_speech_filter: Set[str] = set(["名詞", "動詞"]),
        use_original: bool = True,
    ) -> None:
        self.tagger = MeCab.Tagger()
        self.part_of_speech_filter = part_of_speech_filter
        self.use_original = use_original

    def tokenize(self, s: str) -> List[str]:
        tokens: List[str] = []
        node = self.tagger.parseToNode(s)
        while node:
            features = node.feature.split(",")
            pos = features[0]
            if pos not in self.part_of_speech_filter or pos == "BOS/EOS":
                node = node.next
                continue
            token = node.surface
            if self.use_original:
                orig = features[-3]
                if orig != "*":
                    token = orig
            tokens.append(token)
            node = node.next
        return tokens


class Vocab:
    def __init__(
        self,
        tokens: Iterable[List[str]],
        max_size: Optional[int] = None,
        min_freq: int = 1,
        specials: List[str] = ["<pad>"],
    ) -> None:
        counter = Counter[str]()
        for toks in tokens:
            for tok in toks:
                counter[tok] += 1
        top_n = counter.most_common(max_size)
        self._id2token = specials + [token for token, n in top_n if n >= min_freq]
        self._token2id = {token: i for i, token in enumerate(self._id2token)}

    @property
    def vocab_size(self) -> int:
        return len(self._id2token)

    def token2index(self, token: str) -> Optional[int]:
        return self._token2id.get(token)

    def index2token(self, index: int) -> Optional[str]:
        if 0 <= index < len(self._id2token):
            return self._id2token[index]
        return None

    def encode_bow(self, tokens: List[str]) -> torch.Tensor:
        tensor = torch.zeros(self.vocab_size, dtype=torch.long)
        for token in tokens:
            index = self.token2index(token)
            if index is not None:
                tensor[index] += 1
        return tensor
