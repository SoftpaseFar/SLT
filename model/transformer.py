import torch
import torch.nn as nn
import torch.nn.functional as f


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
        output = self.norm(src)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, src, src_mask):
        # Step 1: Self-Attention Mechanism
        mh_output = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        rc_1 = src + self.dropout(mh_output)  # Residual Connection
        src_1 = self.norm1(rc_1)  # Layer Normalization

        # Step 2: Feedforward Neural Network (FFN)
        ff_output = self.linear2(f.relu(self.linear1(src_1)))
        rc_2 = src_1 + self.dropout(ff_output)  # Residual Connection
        output = self.norm2(rc_2)  # Layer Normalization

        return output


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
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        # Self-Attention Mechanism
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Src-Attention Mechanism
        tgt2 = self.src_attn(tgt, enc_output, enc_output, attn_mask=src_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward Neural Network (FFN)
        tgt2 = self.linear2(f.relu(self.linear1(tgt)))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt
