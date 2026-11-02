import os
import argparse
from src.tokenizer.basic_tokenizer import BaseTokenizer
from src.tokenizer.regex_tokenizer import RegexTokenizer
from datasets import load_from_disk
SPECIAL_TOKENS={'<|endoftext|>': 513}
TEST_TEXT = "The trainer’s own saves do not follow its resume convention <|endoftext|>"
def run(args):

    tiny_ds = load_from_disk(os.path.join(args.data_path, 'TinyStories'))
    wiki_ds = load_from_disk(os.path.join(args.data_path, 'wikitext-103-v1'))
    fineweb_ds = load_from_disk(os.path.join(args.data_path, 'fineweb_edu'))

    tiny_text = "\n".join(tiny_ds["train"][:2000]["text"])
    wiki_text = "\n".join(wiki_ds["train"]["text"])
    fineweb_text = "\n".join(fineweb_ds["train"]["text"])
    print("Length of tiny_ds: ", len(tiny_ds["train"]))
    print("Length of wiki_ds: ", len(wiki_ds["train"]["text"]))
    print("Length of fineweb_ds: ", len(fineweb_ds["train"]["text"]))

    text = "\n".join([tiny_text, wiki_text, fineweb_text])
    if args.tokenizer_type == "regex":
        print("Initializing RegexTokenizer...")
        tokenizer = RegexTokenizer()
    elif args.tokenizer_type == "basic":
        print("Initializing BaseTokenizer...")
        tokenizer = BaseTokenizer()
    else:
        raise ValueError("Invalid tokenizer type. Choose 'regex' or 'basic'.")

    print("Training")
    tokenizer.train(args.vocab_size, text)
    tokenizer.register_special_tokens(SPECIAL_TOKENS)
    # tokenizer.save(f"{args.save_path}/{args.tokenizer_type}")
    print("Testing tokenizer")
    tokenizer.load(f"{args.save_path}/{args.tokenizer_type}")
    ids = tokenizer.encode(TEST_TEXT)
    print("Tokenized ids: ", ids)
    decoded_text = tokenizer.decode(ids)
    print("Ground Truth Text == Decoded Text: ", TEST_TEXT==decoded_text)

def main():
    parser = argparse.ArgumentParser(description="Train a tokenizer.")
    parser.add_argument("--tokenizer_type", type=str, choices=["regex", "basic"], required=True, help="Type of tokenizer to train.")
    parser.add_argument("--vocab_size", type=int, default=512, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to texts for training.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the tokenizer.")
    args = parser.parse_args()

    run(args=args)

if __name__ == "__main__":
    main()