# nanoLM

The purpose of this project is to learn about language models from basics

Currently this repo contains training and inference code for BPE based tokenizers inspired by Andrej Karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) video.

## Notes

For a detailed explanation of BPE tokenization and the ideas behind this implementation, see my [BPE Tokenizer notes](https://riteshrm.github.io/posts/bpe-tokenizer/).

```
nanoLM/
│
├── tokenizer/
│   ├── __init__.py
│   ├── base.py
│   ├── basic_tokenizer.py
│   ├── inference.py
│   ├── regex_tokenizer.py
│   └── train.py
│
├── utils/
│   ├── __init__.py
│   └── data_downloader.py
│
└── requirements.txt
```

## Components

1. [`base.py`](tokenizer/base.py) acts as the base for all the tokenizers.
2. [`basic_tokenizer.py`](tokenizer/basic_tokenizer.py) implements everything needed for a normal tokenizer without considering patterns, i.e. `dog.`, `dog?`, and `dog!` are treated as different tokens, which is not appropriate.
3. [`regex_tokenizer.py`](tokenizer/regex_tokenizer.py) tries to mitigate the issues present in the basic tokenizer by first chunking the given text based on a regex pattern and then performing merging.
4. [`data_downloader.py`](utils/data_downloader.py) is used to download the [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories).
5. [`train.py`](tokenizer/train.py) is responsible for training the tokenizer based on the provided tokenizer type and data path.
6. [`inference.py`](tokenizer/inference.py) is used to verify whether the trained tokenizer is working properly. We provide the saved tokenizer path and tokenizer type, alongside an optional text file. It encodes and decodes the text, then compares the decoded text against the original.

## Setup

Clone the repo and install the packages mentioned in requirements.txt

```bash
git clone https://github.com/riteshrm/nanoLM.git
cd nanoLM
pip install -r requirements.txt
```

## Download Data

This will download the required dataset.

```bash
python -m utils.data_downloader
```

## Train

```bash
python -m tokenizer.train --tokenizer_type="regex" --vocab_size=512 --data_path="data/TinyStories" --save_path="tokenizer_model"
```

The tokenizer type can be either `basic` or `regex`. The vocabulary size defaults to `512`. `data_path` should point to the downloaded TinyStories dataset.

After training, `merges.model` and `vocab.vocab` are saved under `<save_path>/<tokenizer_type>/`.

## Inference

```bash
python -m tokenizer.inference --tokenizer_type="regex" --save_path="tokenizer_model/regex"
```
