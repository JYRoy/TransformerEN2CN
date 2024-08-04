import torch
import tiktoken


class DataLoaderCMN:

    def __init__(self, batch_size, seq_len, split="train"):
        assert split in {"train", "val"}
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.current_position = 0

        self.tokens = []
        text = open("./datasets/cmn.txt").read().strip().split("\n")
        enc = tiktoken.get_encoding("cl100k_base")
        for line in text:
            en, cn, attribute = line.split("\t")
            en = enc.encode(en.lower())
            cn = enc.encode(cn)
            self.tokens.append((en, cn))
        self.num_paris = len(self.tokens)  # 24360 total,

        self.tokens = (
            self.tokens[: self.num_paris * 90 // 100]
            if split == "train"
            else self.tokens[self.num_paris * 90 // 100 + 1 :]
        )
        self.num_tokens = len(self.tokens)  # 24360 total,
        print(f"loaded {self.num_tokens} english to chinese sentence pairs")

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        batch = self.tokens[
            self.current_position : self.current_position + self.batch_size
        ]
        x = [pair[0] for pair in batch]
        y = [pair[1] for pair in batch]
        self.current_position += self.batch_size
        if self.current_position + self.batch_size > self.num_tokens:
            self.current_position = 0
        return x, y


# dataloader = DataLoaderCMN(2, 20)
# dataloader.next_batch()
