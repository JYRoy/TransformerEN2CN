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

    def forward(self, q, k, v, pad_mask=None, subsequent_mask: bool = False):
        batch_size, seq_len, d_model = q.size()

        # 通过三个独立的权重矩阵获得q、k、v
        q = self.q_weight(q)
        k = self.k_weight(k)
        v = self.v_weight(v)

        # 通过view转换分为h个头，每个头的维度是d_model//n_head
        # (batch_size, seq_len, n_head, d_model // n_head) -> (batch_size, n_head, seq_len, d_model // n_head)
        q = q.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )
        k = k.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )
        v = v.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )

        # 计算 attention score
        # score含义是行 i（src的token i）和列 j（target的token j）间的attention score
        # q @ k:
        #       (batch_size, n_head, seq_len, d_model // n_head) @
        #       (batch_size, n_head, d_model // n_head, seq_len) =
        #       (batch_size, n_head, q_seq_len, k_seq_len)
        # q_seq_len, k_seq_len）中[i, j]位置表示q中的第i个token和k中的第j个token的attention score
        #
        # 对于cross attention：
        #   q来自于decoder上一个block， k来自于encoder
        scores = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        # mask掉padding的token，避免pad参与attention运算
        if pad_mask != None:
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))
        # 用于decoder中的第一个attention block，mask掉当前token后面的token
        if subsequent_mask:
            scores = scores.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
        scores = F.softmax(scores, dim=-1)

        y = scores @ v
        # 把各个head拼在一起
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

    def forward(self, x, src_mask):
        x = self.layer_norm_1(x + self.attention(x, x, x, src_mask))
        x = self.layer_norm_2(x + self.feed_forward(x))
        return x


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mask_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.layer_norm_1 = LayerNorm(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = LayerNorm(config)
        self.layer_norm_3 = LayerNorm(config)

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        x = self.layer_norm_1(x + self.mask_attention(x, x, x, tgt_mask, True))

        x = self.layer_norm_2(
            x + self.cross_attention(x, encoder_output, encoder_output, src_mask)
        )
        x = self.layer_norm_3(x + self.feed_forward(x))
        return x


class TransformerInput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 按照最长的输入长度初始化位置编码矩阵，[max_seq_len, d_model]
        self.position_embedding = torch.zeros(config.max_seq_len, config.d_model).to(
            "cuda"
        )
        position = torch.arange(0, config.max_seq_len)
        # 增加embedding维度
        position = position.unsqueeze(1)

        i = torch.arange(0, config.d_model // 2, 1)
        div = 10000 ** (2 * i / config.d_model)
        # [max_seq_len, d_model // 2]
        term = position / div
        # embedding的偶数位置是sin
        self.position_embedding[:, 0::2] = torch.sin(term)
        # embedding的奇数位置是cos
        self.position_embedding[:, 1::2] = torch.cos(term)
        # 注册为buffer，不参与训练
        self.register_buffer("pe", self.position_embedding)

    def forward(self, x):
        # x: [batch_size, seq_len]
        # [batch_size, seq_len, d_model] + [seq_len, d_model]
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

    def forward(self, source, target, src_mask, tgt_mask):
        enc_out = self.encode(source, src_mask)
        target = self.decode(target, enc_out, tgt_mask, src_mask)
        return self.output(target)

    def encode(self, source, src_mask):
        x = self.src_input(source)
        for encoder in self.encoders:
            x = encoder(x, src_mask)
        return x

    def decode(self, target, enc_out, tgt_mask, src_mask):
        target = self.tgt_input(target)
        for decoder in self.decoders:
            target = decoder(target, enc_out, tgt_mask, src_mask)
        return target
