import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import *
from model import *

BATCH_SIZE = 8
train_data_loader = DataLoaderCMN(batch_size=BATCH_SIZE, split="train")
val_data_loader = DataLoaderCMN(batch_size=BATCH_SIZE, split="val")

config = TransformerConfig(
    train_max_seq_len=train_data_loader.max_seq_len,
    val_max_seq_len=val_data_loader.max_cn_seq_len,
)

model = TransformerModel(config=config)
model.to("cuda")
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p)

# model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)


def train(epochs, print_every=100):
    model.train()
    start = time.time()
    temp = start
    total_loss = 0
    for epoch in range(epochs):
        for step in range(train_data_loader.steps_per_epoch):
            optimizer.zero_grad()
            x, y = train_data_loader.next_batch()
            x, y = x.to("cuda"), y.to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x, y)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=tiktoken.get_encoding("cl100k_base").eot_token,
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % print_every == 0:
                loss_avg = total_loss / print_every
                print(
                    "time = %ds, epoch %d, iter = %d, loss = %.3f, %ds per %d iters"
                    % (
                        (time.time() - start),
                        epoch + 1,
                        step + 1,
                        loss_avg,
                        time.time() - temp,
                        print_every,
                    )
                )
                total_loss = 0
                temp = time.time()


def translate():
    model.eval()
    enc = tiktoken.get_encoding("cl100k_base")
    src, target = val_data_loader.next_batch()
    src, target = src.to("cuda"), target.to("cuda")
    outputs = torch.zeros(100).type_as(src.data)
    print(enc.decode(src[0, :].cpu().numpy().tolist()))
    print(enc.decode(target[0, :].cpu().numpy().tolist()))
    for step in range(1, config.val_max_seq_len):
        e_out = model.encode(src)
        out = model.decode(e_out, outputs[:step].unsqueeze(0))
        out = model.output(out)
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)
        outputs[step] = ix[0][0].item()
    print(enc.decode(outputs[1:].cpu().numpy().tolist()))


if __name__ == "__main__":
    train(1, 100)
    translate()
