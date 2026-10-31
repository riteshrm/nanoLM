import argparse
from .basic_tokenizer import BaseTokenizer
from .regex_tokenizer import RegexTokenizer
DEFAULT_TEST_TEXT  = "This is a test text for testing the tokenizer. Let's see how it works!"

def run(args):
    if args.tokenizer_type == "regex":
        print("Infering RegexTokenizer...")
        tokenizer = RegexTokenizer()
    elif args.tokenizer_type == "basic":
        print("Infering BaseTokenizer...")
        tokenizer = BaseTokenizer()
    else:
        raise ValueError("Invalid tokenizer type. Choose 'regex' or 'basic'.")
    tokenizer.load(args.save_path)

    test_text = DEFAULT_TEST_TEXT 
    if args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            test_text = f.read()
    tokens = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(tokens)
    assert test_text == decoded_text, "Decoded text does not match the original text."
    print("Decoded text and original text are same")


def main():
    parser = argparse.ArgumentParser(description="Train a tokenizer.")
    parser.add_argument("--tokenizer_type", type=str, choices=["regex", "basic"], required=True, help="Type of tokenizer to train.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to load the trained tokenizer from.")
    parser.add_argument("--text_file", type=str, required=False, help="Path to the text file for testing.")
    args = parser.parse_args()

    run(args=args)

if __name__ == "__main__":
    main()