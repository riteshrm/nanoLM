# from datasets import load_dataset

# ds = load_dataset("roneneldan/TinyStories")

# ds.save_to_disk("data/TinyStories")

from datatrove.pipeline.readers import ParquetReader
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT") 

import json
total_tokens = 0
target_tokens = 400e6

with open("fineweb_edu.jsonl", "w") as f:
    for doc in data_reader():
        if doc.metadata["score"] < 4:
            continue

        text = doc.text
        total_tokens += doc.metadata["token_count"]

        f.write(json.dumps({"text": text}) + "\n")

        if total_tokens >= target_tokens:
            break

print(f"Collected approximately {total_tokens:,} tokens")