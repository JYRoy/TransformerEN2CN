import random
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_scheduler
from sacrebleu.metrics import BLEU


from dataloader import *
from model import *

seed = 2024
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
BATCH_SIZE = 8
EPOCH_NUM = 5

data = CMNDataset(data_file="datasets/cmn.txt", max_dataset_size=10000)
train_data, valid_data, test_data = random_split(data, [8000, 1000, 1000])
print("tokenizer.vocab_size: ", tokenizer.vocab_size)

config = TransformerConfig(vocab_size=tokenizer.vocab_size + 1)

model = TransformerModel(config=config)
model.to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p)

# model = torch.compile(model)

train_dataloader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collote_fn
)
valid_dataloader = DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collote_fn
)
test_dataloader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collote_fn
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCH_NUM * len(train_dataloader),
)


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        x = batch_data["input_ids"].to(device)
        y = batch_data["labels"].to(device)
        src_mask = batch_data["attention_mask"].unsqueeze(-2).to(device)
        tgt_mask = batch_data["tgt_mask"].unsqueeze(-2).to(device)
        logits = model(x, y, src_mask, tgt_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=65000
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(
            f"loss: {total_loss/(finish_batch_num + batch):>7f}"
        )
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model):
    preds, labels = [], []

    model.eval()
    for batch_data in tqdm(dataloader):
        x = batch_data["input_ids"].to(device)
        y = batch_data["labels"].to(device)
        src_mask = batch_data["attention_mask"].unsqueeze(-2).to(device)

        outputs = torch.zeros(BATCH_SIZE, 100).type_as(x.data)
        outputs[0] = 65001
        with torch.no_grad():
            for step in range(1, 100):
                e_out = model.encode(x, src_mask)
                tgt_mask = torch.tril(torch.ones((BATCH_SIZE, step, step))).to(device)
                out = model.decode(outputs[:, :step], e_out, tgt_mask, src_mask)
                out = model.output(out)
                out = F.softmax(out, dim=-1)
                val, ix = out[:, :, -1].data.topk(1)
                outputs[:, step] = ix[:][0].item()

        with tokenizer.as_target_tokenizer():
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        label_tokens = batch_data["labels"].cpu().numpy()
        label_tokens = np.where(
            label_tokens != 65000, label_tokens, tokenizer.pad_token_id
        )
        with tokenizer.as_target_tokenizer():
            decoded_labels = tokenizer.batch_decode(
                label_tokens, skip_special_tokens=True
            )

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    bleu_score = bleu.corpus_score(preds, labels).score
    print("bleu_score: ", bleu_score)
    print(preds[0])
    print(labels[0])


bleu = BLEU()
total_loss = 0.0
for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}/{EPOCH_NUM}\n-------------------------------")
    total_loss = train_loop(
        train_dataloader, model, optimizer, lr_scheduler, t + 1, total_loss
    )
    test_loop(valid_dataloader, model)

print("Done!")
