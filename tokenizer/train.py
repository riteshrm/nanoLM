import argparse
from .basic_tokenizer import BaseTokenizer
from .regex_tokenizer import RegexTokenizer
from datasets import load_from_disk

def run(args):

    ds = load_from_disk(args.data_path)
    text = "\n".join(ds["train"][:20000]["text"])
    if args.tokenizer_type == "regex":
        print("Training RegexTokenizer...")
        tokenizer = RegexTokenizer()
    elif args.tokenizer_type == "basic":
        print("Training BaseTokenizer...")
        tokenizer = BaseTokenizer()
    else:
        raise ValueError("Invalid tokenizer type. Choose 'regex' or 'basic'.")

    tokenizer.train(args.vocab_size, text)
    tokenizer.save(f"{args.save_path}/{args.tokenizer_type}")

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