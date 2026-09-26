
from .base import get_stats, merge, Tokenizer
    

class BaseTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def encode(self, text):
        tokens = text.encode('utf-8')
        ids = list(map(int,tokens))
        while len(ids)>=2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges.get(pair)
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        tokens = b''.join([self.vocab[id] for id in ids])
        text = tokens.decode('utf-8', errors='replace')
        return text

    def train(self, vocab_size, text):
        assert vocab_size >= 256, "vocab_size must be greater than 256"
        self.vocab_size = vocab_size
        num_merges = self.vocab_size - 256
        merges = {}
        tokens = text.encode('utf-8')
        ids = list(map(int, tokens))
        assert len(ids) >= 2, "Input text must contain at least two bytes."
        for i in range(num_merges):
            stats = get_stats(ids)
            if len(stats)==0:
                print("No more pairs to merge. Stopping training.")
                break
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, top_pair, idx)
            merges[top_pair] = idx
        self.merges = merges
        self.vocab = self.build_vocab()
        self.vocab_size = len(self.vocab)