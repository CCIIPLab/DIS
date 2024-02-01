# -*- coding: utf-8 -*-

import os
import sys
root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print("append project dir {} to environment".format(root_dir))
sys.path.append(root_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import PositionalEmbedding
from model.modules import EncoderLayer, DecoderLayer
from model.utils import get_non_pad_mask, get_attn_key_pad_mask, get_subsequent_mask
from scripts import Constants

class TransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self, vocab_size,
                 d_word_vec,
                 n_layers,
                 d_model,
                 n_head,
                 dropout=0.1):
        super(TransformerDecoder, self).__init__()

        d_k = d_model // n_head
        d_v = d_model // n_head
        d_inner = d_model * 4

        self.word_emb = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
        self.src_proj = nn.Linear(300, d_word_vec)

        self.post_word_emb = PositionalEmbedding(d_model=d_word_vec)

        self.enc_layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt_seq, src_seq):
        # -- Encode source
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)

        enc_inp = self.src_proj(self.word_emb(src_seq)) + self.post_word_emb(src_seq)

        for enc_layer in self.enc_layer_stack:
            enc_inp, _ = enc_layer(enc_inp, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        enc_output = enc_inp

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.src_proj(self.word_emb(tgt_seq)) + self.post_word_emb(tgt_seq)
        #dec_output += vis_feat.unsqueeze(1)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                               non_pad_mask=non_pad_mask,
                                                               slf_attn_mask=slf_attn_mask,
                                                               dec_enc_attn_mask=dec_enc_attn_mask)

        logits = self.tgt_word_prj(dec_output)
        return logits

    def translate_batch(self, de_vocab, src_seq, max_token_seq_len=30):
        with torch.no_grad():
            # -- Encode source
            non_pad_mask = get_non_pad_mask(src_seq)
            slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)

            enc_inp = self.src_proj(self.word_emb(src_seq)) + self.post_word_emb(src_seq)

            for layer in self.enc_layer_stack:
                enc_inp, _ = layer(enc_inp, non_pad_mask, slf_attn_mask)
            enc_output = enc_inp

            trg_seq = torch.full((src_seq.size(0), 1), Constants.SOS, dtype=torch.long, device=src_seq.device)

            dec_outputs = []
            for len_dec_seq in range(0, max_token_seq_len + 1):
                # -- Prepare masks
                non_pad_mask = get_non_pad_mask(trg_seq)
                slf_attn_mask_subseq = get_subsequent_mask(trg_seq)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=trg_seq, seq_q=trg_seq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=trg_seq)

                dec_output = self.src_proj(self.word_emb(trg_seq)) + self.post_word_emb(trg_seq)
                #dec_output += vis_feat.unsqueeze(1)

                if len_dec_seq == 0:
                    dec_outputs.append(dec_output)
                else:
                    dec_outputs[0] = dec_output

                for i, dec_layer in enumerate(self.layer_stack):
                    tmp = dec_layer.step_forward(
                        dec_outputs[i], enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=dec_enc_attn_mask)

                    if i == len(self.layer_stack) - 1:
                        break
                    else:
                        if len_dec_seq == 0:
                            dec_outputs.append(tmp)
                        else:
                            dec_outputs[i + 1] = torch.cat([dec_outputs[i + 1], tmp], 1)

                logits = self.tgt_word_prj(tmp.squeeze(1))

                chosen = torch.argmax(logits, -1)
                trg_seq = torch.cat([trg_seq, chosen.unsqueeze(-1)], -1)

        result = []
        for _ in trg_seq:
            result.append("")
            for elem in _:
                if elem.item() == Constants.EOS:
                    break
                elif elem.item() == Constants.SOS:
                    continue
                else:
                    result[-1] += de_vocab[elem.item()]
        return result