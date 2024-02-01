# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, LayerNorm

class AttFlat(nn.Module):
    def __init__(self, hidden_size, out_size, dropout):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=hidden_size,
            out_size=1,
            dropout_r=dropout,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size,
            out_size
        )

    def forward(self, x, x_mask=None):
        att = self.mlp(x)

        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                float("-inf")
            )
        att = F.softmax(att, dim=1)  # [B, N, 1]

        x_atted = (att * x).sum(1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class MHAtt(nn.Module):
    def __init__(self, head_num, hidden_size, dropout, hidden_size_head):
        super(MHAtt, self).__init__()
        self.head_num = head_num
        self.hidden_size = hidden_size
        self.hidden_size_head = hidden_size_head
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

class SA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(SA, self).__init__()

        self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        output = self.mhatt(x, x, x, x_mask)
        dropout_output = self.dropout1(output)
        x = self.norm1(x + dropout_output)

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x

class SGA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.mhatt2 = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class GA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(GA, self).__init__()

        self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, y, y_mask, x_mask=None):
        if x_mask is None:
            intermediate = self.dropout1(self.mhatt(y, y, x, y_mask))
        else:
            intermediate = self.dropout1(self.mhatt(y, y, x, y_mask)) * x_mask.unsqueeze(-1)

        x = self.norm1(x + intermediate)
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

    def step_forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_query = dec_input[:, -1, :].unsqueeze(1)
        slf_attn_mask = slf_attn_mask[:, -1, :].unsqueeze(1)
        dec_enc_attn_mask = dec_enc_attn_mask[:, -1, :].unsqueeze(1)
        non_pad_mask = non_pad_mask[:, -1, :].unsqueeze(1)

        dec_output, dec_slf_attn = self.slf_attn(
            dec_query, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output