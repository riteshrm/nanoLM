
from .base import get_stats, merge
from .basic_tokenizer import BaseTokenizer
import regex as re
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BaseTokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = pattern
        self.compiled_pattern = re.compile(self.pattern)

    def encode(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for text_chunk in text_chunks:
            ids.extend(super().encode(text_chunk))
        return ids

    def train(self, vocab_size, text):
        assert vocab_size >= 256, "vocab_size must be greater than 256"
        self.vocab_size = vocab_size
        num_merges = self.vocab_size - 256
        merges = {}
        text_chunks = re.findall(self.compiled_pattern, text)
        tokens = [list(chunk.encode('utf-8')) for chunk in text_chunks]
        ids = list(tokens)
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                if len(chunk_ids) >= 2:
                    stats = get_stats(chunk_ids, stats)
            if len(stats)==0:
                print("No more pairs to merge. Stopping training.")
                break
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            merges[top_pair] = idx
        self.merges = merges
        self.vocab = self.build_vocab()
        self.vocab_size = len(self.vocab)

    def load(self, filepath):
        super().load(filepath)
        self.compiled_pattern = re.compile(self.pattern)