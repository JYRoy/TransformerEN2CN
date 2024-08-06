from dataloader import *
from model import *
import torch
import torch.nn as nn


BATCH_SIZE = 2
MAX_STEPS = 50
train_data_loader = DataLoaderCMN(batch_size=BATCH_SIZE, split="train")
val_data_loader = DataLoaderCMN(batch_size=BATCH_SIZE, split="val")

config = TransformerConfig(max_seq_len=train_data_loader.max_seq_len)

model = TransformerModel(config=config)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
loss_compute = nn.CrossEntropyLoss().to("cuda")
model.train()
model.to("cuda")
for step in range(MAX_STEPS):
    optimizer.zero_grad()
    x, y = train_data_loader.next_batch()
    x, y = x.to("cuda"), y.to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x, y)

    loss = loss_compute(logits.view(-1, logits.size(-1)), y.view(-1))
    print(loss)
    loss.backward()
    optimizer.step()
