import torch
import tiktoken

import numpy as np


def seq_padding(batch, padding=0):
    L = [len(seq) for seq in batch]
    max_len = max(L)
    return np.array(
        [
            (
                np.concatenate([seq, [padding] * (max_len - len(seq))])
                if len(seq) < max_len
                else seq
            )
            for seq in batch
        ]
    )


# 按照句子长度排序，尽可能减少padding
def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))


class DataLoaderCMN:

    def __init__(self, batch_size, split="train"):
        assert split in {"train", "val"}
        self.batch_size = batch_size
        self.current_position = 0

        self.tokens = []
        self.en_tokens = []
        self.cn_tokens = []
        text = open("./datasets/cmn.txt").read().strip().split("\n")

        self.enc = tiktoken.get_encoding("cl100k_base")

        for line in text:
            en, cn, attribute = line.split("\t")
            en = self.enc.encode(en.lower())
            cn = self.enc.encode(cn)
            self.tokens.append((en, cn))
            self.en_tokens.append(en)
            self.cn_tokens.append(cn)
        self.num_paris = len(self.tokens)  # 24360 total
        print(f"loaded {self.num_paris} english to chinese sentence pairs")

        sorted_index = len_argsort(self.en_tokens)
        self.en_tokens = [self.en_tokens[i] for i in sorted_index]
        self.cn_tokens = [self.cn_tokens[i] for i in sorted_index]
        self.en_tokens = (
            self.en_tokens[: self.num_paris * 90 // 100]
            if split == "train"
            else self.en_tokens[self.num_paris * 90 // 100 + 1 :]
        )

        self.cn_tokens = (
            self.en_tokens[: self.num_paris * 90 // 100]
            if split == "train"
            else self.cn_tokens[self.num_paris * 90 // 100 + 1 :]
        )

        self.max_en_seq_len = max(len(s) for s in self.en_tokens)
        self.max_cn_seq_len = max(len(s) for s in self.cn_tokens)
        self.max_seq_len = max(self.max_en_seq_len, self.max_cn_seq_len)

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        x_batch = self.en_tokens[
            self.current_position : self.current_position + self.batch_size
        ]
        y_batch = self.cn_tokens[
            self.current_position : self.current_position + self.batch_size
        ]

        self.current_position += self.batch_size
        if self.current_position + self.batch_size > self.num_paris:
            self.current_position = 0
        return seq_padding(x_batch, self.enc.eot_token), seq_padding(
            y_batch, self.enc.eot_token
        )
