from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

ds.save_to_disk("data/TinyStories")

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

ds.save_to_disk("data/wikitext-103-v1")

from datatrove.pipeline.readers import ParquetReader

# data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT") 

# import json
# total_tokens = 0
# target_tokens = 400000000

# with open("data/fineweb_edu.jsonl", "w") as f:
#     for doc in data_reader():
#         if doc.metadata["score"] < 4:
#             continue

#         text = doc.text
#         total_tokens += doc.metadata["token_count"]

#         f.write(json.dumps({"text": text}) + "\n")

#         if total_tokens >= target_tokens:
#             break

# print(f"Collected approximately {total_tokens:,} tokens")

import random

random.seed(42)

with open("data/fineweb_edu.jsonl", "r") as src, \
     open("data/fineweb_edu_train.jsonl", "w") as train, \
     open("data/fineweb_edu_val.jsonl", "w") as val:

    for line in src:
        if random.random() < 0.95:
            train.write(line)
        else:
            val.write(line)

ds = load_dataset(
    "json",
    data_files={
        "train": "data/fineweb_edu_train.jsonl",
        "validation": "data/fineweb_edu_val.jsonl"
    }
)

ds.save_to_disk("data/fineweb_edu")
