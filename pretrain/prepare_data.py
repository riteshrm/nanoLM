import argparse
from tokenizer.basic_tokenizer import BaseTokenizer
from tokenizer.regex_tokenizer import RegexTokenizer
from datasets import load_from_disk
import numpy as np

def run(args):

    ds = load_from_disk(args.data_path)
    train_text = "\n".join(ds["train"][:50000]["text"])
    val_text = "\n".join(ds["validation"][:1000]["text"])
    if args.tokenizer_type == "regex":
        print("Using RegexTokenizer...")
        tokenizer = RegexTokenizer()
    elif args.tokenizer_type == "basic":
        print("Using BaseTokenizer...")
        tokenizer = BaseTokenizer()
    else:
        raise ValueError("Invalid tokenizer type. Choose 'regex' or 'basic'.")

    print("Loading Tokenizer...")
    tokenizer.load(f"{args.tokenizer_path}/{args.tokenizer_type}")

    ##Trin data
    print("Encoding train data")
    encoded_tokens = tokenizer.encode(text=train_text)
    tokens = np.asarray(encoded_tokens, dtype=np.uint32)
    np.save(f"{args.save_path}/{args.tokenizer_type}_pretrain_train.npy",tokens)

    print("Encoding validation data")
    ## Validation data
    encoded_tokens = tokenizer.encode(text=val_text)
    tokens = np.asarray(encoded_tokens, dtype=np.uint32)
    np.save(f"{args.save_path}/{args.tokenizer_type}_pretrain_val.npy",tokens)

    

def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--tokenizer_type", type=str, choices=["regex", "basic"], required=True, help="Type of tokenizer to train.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to texts for training.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the encoded data.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to load the tokenizer.")
    args = parser.parse_args()

    run(args=args)

if __name__ == "__main__":
    main()

# python -m pretrain.prepare_data --tokenizer_type='regex' --data_path="data/TinyStories" --save_path="data" --tokenizer_path="tokenizer_model"