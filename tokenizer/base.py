import unicodedata
from pathlib import Path
import os

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair, 0)+1
    return counts

def merge(ids, pair, token_number):
    new_ids = []
    i=0
    while i< len(ids):
        if i+1<len(ids) and ids[i]==pair[0] and ids[i+1]==pair[1]:
            new_ids.append(token_number)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# This function basically to explicitly show the control characters in tokens.
def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

# To go from bytes to string
def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s
class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self.build_vocab()

    def encode(self, text):
        raise NotImplementedError("The encode method should be implemented in a subclass.")

    def decode(self, ids):
        raise NotImplementedError("The decode method should be implemented in a subclass.")

    def train(self, vocab_size, text):
        raise NotImplementedError("The train method should be implemented in a subclass.")

    def save(self, filepath):
            os.makedirs(filepath, exist_ok=True)
            merges_file = Path(filepath, "merges.model")

            with open(merges_file, "w") as f:
                f.write(f"{self.pattern}\n")
                f.write(f"{len(self.special_tokens)}\n")
                for special, idx in self.special_tokens.items():
                    f.write(f"{special} {idx}\n")
                for pair, idx in self.merges.items():
                    f.write(f"{pair[0]} {pair[1]} {idx}\n")
            vocab_file = Path(filepath, "vocab.vocab")
            inverted_merges = {v: k for k, v in self.merges.items()}
            with open(vocab_file, "w") as f:
                for idx, token in self.vocab.items():
                    s = render_token(token)
                    if idx in inverted_merges: # If current token has child tokens, then render it as pair
                        pair = inverted_merges[idx]
                        s0 = render_token(self.vocab[pair[0]])
                        s1 = render_token(self.vocab[pair[1]])
                        token_str = f"[{s0} {s1}] -> [{s}] {idx}"
                    else:
                        token_str = f"[{s}] {idx}"
                    f.write(f"{token_str}\n")
    def load(self, filepath):
        merges_file = Path(filepath, "merges.model")
        with open(merges_file, "r") as f:
            self.pattern = f.readline().strip()
            num_special_tokens = int(f.readline().strip())
            special_tokens = {}
            for _ in range(num_special_tokens):
                line = f.readline().strip()
                special, idx = line.split()
                special_tokens[special] = int(idx)
            merges = {}
            for line in f:
                idx1, idx2, idx = line.strip().split()
                pair = (int(idx1), int(idx2))
                idx = int(idx)
                merges[pair] = idx
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self.build_vocab()

    def build_vocab(self):
        vocab={idx:bytes([idx]) for idx in range(256)}
        
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode('utf-8')
        return vocab