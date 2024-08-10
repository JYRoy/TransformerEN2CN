import copy
import inspect
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 65001
    train_max_seq_len: int = 1024
    val_max_seq_len: int = 1024
    max_seq_len: int = max(train_max_seq_len, val_max_seq_len)
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 768


class LayerNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.a = nn.Parameter(torch.ones(config.d_model))
        self.b = nn.Parameter(torch.zeros(config.d_model))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        y = self.a * (x - mean) / (std + self.eps) + self.b
        return y


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.q_weight = nn.Linear(config.d_model, config.d_model)
        self.k_weight = nn.Linear(config.d_model, config.d_model)
        self.v_weight = nn.Linear(config.d_model, config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.n_head = config.n_head
        self.d_model = config.d_model

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )

    def forward(self, q, k, v, mask: bool = False):
        batch_size, seq_len, d_model = q.size()

        q = self.q_weight(q)
        k = self.k_weight(k)
        v = self.v_weight(v)

        q = q.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )
        k = k.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )
        v = v.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )

        scores = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        if mask:
            scores = scores.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
        scores = F.softmax(scores, dim=-1)

        y = scores @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        y = self.proj(y)
        return y


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm_1 = LayerNorm(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = LayerNorm(config)

    def forward(self, x):
        x = self.layer_norm_1(x + self.attention(x, x, x))
        x = self.layer_norm_2(x + self.feed_forward(x))
        return x


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mask_attention = MultiHeadAttention(config)
        self.attention = MultiHeadAttention(config)
        self.layer_norm_1 = LayerNorm(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = LayerNorm(config)
        self.layer_norm_3 = LayerNorm(config)

    def forward(self, x, encoder_output):
        x = self.layer_norm_1(x + self.mask_attention(x, x, x, True))
        x = self.layer_norm_1(x + self.attention(x, encoder_output, encoder_output))
        x = self.layer_norm_1(x + self.feed_forward(x))
        return x


class TransformerInput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.position_embedding = torch.zeros(config.max_seq_len, config.d_model).to(
            "cuda"
        )
        position = torch.arange(0, config.max_seq_len)
        position = position.unsqueeze(1)

        i = torch.arange(0, config.d_model // 2, 1)
        div = 10000 ** (2 * i / config.d_model)
        term = position / div
        self.position_embedding[:, 0::2] = torch.sin(term)
        self.position_embedding[:, 1::2] = torch.cos(term)

        self.register_buffer("pe", self.position_embedding)

    def forward(self, x):
        return self.token_embedding(x) + self.position_embedding[: x.size(1), :]


class TransformerOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.head(x), dim=-1)


class TransformerModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.src_input = TransformerInput(config)
        self.tgt_input = TransformerInput(config)
        self.encoders = nn.ModuleList(
            [copy.deepcopy(Encoder(config)) for _ in range(config.n_layer)]
        )
        self.decoders = nn.ModuleList(
            [copy.deepcopy(Decoder(config)) for _ in range(config.n_layer)]
        )
        self.output = TransformerOutput(config)

    def forward(self, source, target):
        enc_out = self.encode(source)
        target = self.decode(enc_out, target)
        output = self.output(target)
        return output

    def encode(self, source):
        x = self.src_input(source)
        for encoder in self.encoders:
            x = encoder(x)
        return x

    def decode(self, enc_out, target):
        target = self.tgt_input(target)
        for decoder in self.decoders:
            target = decoder(target, enc_out)
        return target
