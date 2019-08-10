from typing import List, Set
import abc

import MeCab


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
            word = node.surface
            if self.use_original:
                orig = features[-3]
                if orig != "*":
                    word = orig
            tokens.append(word)
            node = node.next
        return tokens
