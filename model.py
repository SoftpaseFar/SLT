import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers, num_heads, dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.linear(dec_output)
        return output


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        src = self.norm(src)
        return src


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, src, src_mask):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout(self.linear(src2))
        src = self.norm(src)
        return src


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(output_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        tgt = self.norm(tgt)
        return tgt


class DecoderLayer(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.src_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(self.linear1(tgt2))
        tgt = self.norm(tgt)
        tgt2 = self.src_attn(tgt, enc_output, enc_output, attn_mask=src_mask)[0]
        tgt = tgt + self.dropout(self.linear2(tgt2))
        tgt = self.norm(tgt)
        return tgt
