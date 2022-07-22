import torch
from torch import nn

from collections import OrderedDict
from typing import Optional


class TransformerMlp(nn.Module):
    def __init__(self, dim, dropout_prob, fc_dims):
        super(TransformerMlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, fc_dims),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_dims, dim),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):

        return self.net(x)



class EncoderBlock(nn.Module):
    def __init__(self, dim, dropout_prob, attn_heads, fc_dims):
        super(EncoderBlock, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=attn_heads,
            dropout=dropout_prob,
            bias=True
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.mlp = TransformerMlp(dim, dropout_prob, fc_dims)


    def forward(self, x):
        h = self.norm(x)
        h, attn_weights = self.attn(h, h, h)
        h = h + x
        h2 = self.norm(h)
        h2 = self.mlp(h2)
        h2 = h2 + h
        return h2




class Transformer(nn.Module):
    def __init__(
                    self,
                    dim: int,
                    attn_heads: int,
                    dropout_prob: float,
                    n_enc_blocks: int,

    ):
        """

        @param dim: embedding dimension size of Transformer encoding block
        @param attn_heads: number of multi-head attention units
        @param dropout_prob: dropout probability
        @param n_enc_blocks: number of stacked encoder blocks

        """
        super(Transformer, self).__init__()


        self.encoder = self._build_encoders(n_enc_blocks, dim, dropout_prob, attn_heads)
        self.dropout = nn.Dropout(dropout_prob)


    def _build_encoders(self, n_enc_blocks, dim, dropout_prob, attn_heads):
        """
        Build stacked encoders
        @param n_enc_blocks:
        @param dropout_prob:
        @param attn_heads:
        @return:
        """
        enc_blocks = []
        for i in range(n_enc_blocks):
            enc_blocks.append(
                (f"encoder_{i}", EncoderBlock(dim, dropout_prob, attn_heads, dim))
            )
        encoder = nn.Sequential(OrderedDict(
            enc_blocks
        ))
        return encoder

    def forward(self, x):
        """
        @param x: (batch size, n objects, embedding dim)
        @return:
        """
        """if self.use_global:
            h_global = self.h_global.expand(x.shape[0], x.shape[-1])
            h_global = h_global.unsqueeze(1)
            x = torch.cat([h_global, x], dim=1)"""
        x = x.transpose(0, 1)
        # needs (sequence length, batch size, embedding dimension)
        h = self.encoder(x)
        return h.transpose(0, 1)