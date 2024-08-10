import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tiktoken
import numpy as np

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

max_input_length = 1024
max_target_length = 1024

model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(["<bos>"])


class CMNDataset(Dataset):

    def __init__(self, data_file=None, max_dataset_size=10000):
        assert data_file != None
        self.max_dataset_size = max_dataset_size
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt") as f:
            for idx, line in enumerate(f):
                if idx >= self.max_dataset_size:
                    break
                en, cn, attribute = line.split("\t")
                cn = "<bos>" + cn
                Data[idx] = {"english": en, "chinese": cn}
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample["english"])
        batch_targets.append(sample["chinese"])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        batch_data["labels"] = labels
    return batch_data


# data = CMNDataset(data_file="datasets/cmn.txt", max_dataset_size=10000)
# train_data, valid_data, test_data = random_split(data, [8000, 1000, 1000])
# print(f"train set size: {len(train_data)}")
# print(f"valid set size: {len(valid_data)}")
# print(f"test set size: {len(test_data)}")
# print(next(iter(train_data)))

# train_dataloader = DataLoader(
#     train_data, batch_size=4, shuffle=True, collate_fn=collote_fn
# )
# valid_dataloader = DataLoader(
#     valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn
# )

# batch = next(iter(train_dataloader))
# print(batch.keys())
# print("batch shape:", {k: v.shape for k, v in batch.items()})
# print(batch)
