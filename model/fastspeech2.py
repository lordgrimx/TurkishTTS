import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class FFTBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        
        # Linear projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)

        # Reshape and concat
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(torch.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class FastSpeech2(nn.Module):
    def __init__(self, max_seq_len, phoneme_vocab_size, encoder_dim, encoder_n_layer, encoder_head, 
                 encoder_conv1d_filter_size, encoder_conv1d_kernel_size, decoder_dim, decoder_n_layer, 
                 decoder_head, decoder_conv1d_filter_size, decoder_conv1d_kernel_size, n_mel_channels):
        super(FastSpeech2, self).__init__()
        
        # Embedding katmanı
        self.embedding = nn.Embedding(phoneme_vocab_size, encoder_dim)
        
        # Position encoding
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(max_seq_len, encoder_dim).unsqueeze(0),
            requires_grad=False
        )
        
        # Encoder
        self.encoder = nn.ModuleList([
            FFTBlock(
                encoder_dim,
                encoder_conv1d_filter_size,
                encoder_head,
                encoder_dim // encoder_head,
                encoder_dim // encoder_head
            ) for _ in range(encoder_n_layer)
        ])
        
        # Length regulator
        self.length_regulator = LengthRegulator()
        
        # Decoder
        self.decoder = nn.ModuleList([
            FFTBlock(
                decoder_dim,
                decoder_conv1d_filter_size,
                decoder_head,
                decoder_dim // decoder_head,
                decoder_dim // decoder_head
            ) for _ in range(decoder_n_layer)
        ])
        
        # Mel-spec projector
        self.mel_linear = nn.Linear(decoder_dim, n_mel_channels)

    def forward(self, src_seq, src_len, duration_target=None, mel_len=None):
        batch_size = src_seq.shape[0]
        max_src_len = src_seq.shape[1]
        
        # Word embedding
        x = self.embedding(src_seq)
        
        # Position encoding
        position_embed = self.position_enc[:, :max_src_len].expand(batch_size, -1, -1)
        
        # Encoder
        x = x + position_embed
        for enc_layer in self.encoder:
            x, _ = enc_layer(x)
        encoder_output = x
        
        # Length regulation
        if self.training and duration_target is not None:
            output, mel_len = self.length_regulator(encoder_output, duration_target, mel_len)
        else:
            output = encoder_output
            
        # Decoder position encoding
        max_mel_len = output.shape[1]
        if max_mel_len > self.position_enc.shape[1]:  # Position encoding boyutunu kontrol et
            # Pozisyon kodlamasını genişlet
            extra_len = max_mel_len - self.position_enc.shape[1]
            extra_pos_enc = get_sinusoid_encoding_table(extra_len, self.position_enc.shape[-1]).unsqueeze(0)
            extra_pos_enc = extra_pos_enc.to(self.position_enc.device)
            self.position_enc = nn.Parameter(
                torch.cat([self.position_enc, extra_pos_enc], dim=1),
                requires_grad=False
            )
        
        decoder_position = self.position_enc[:, :max_mel_len].expand(batch_size, -1, -1)
        
        # Decoder
        decoder_output = output + decoder_position
        for dec_layer in self.decoder:
            decoder_output, _ = dec_layer(decoder_output)
        
        # Mel-spec projection
        mel_output = self.mel_linear(decoder_output)
        
        return mel_output, None, None, None

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(1)).bool()
    
    return mask

class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, duration_target, target_mel_len=None):
        output = list()
        mel_len = list()
        
        for batch, expand_target in zip(x, duration_target):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        # Hedef mel uzunluğuna göre padding yap
        if target_mel_len is not None:
            max_len = target_mel_len.max().item()
            output = pad_1d_tensor(output, max_len)
        else:
            output = pad_1d_tensor(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 1), -1))
        out = torch.cat(out, 0)
        return out

def pad_1d_tensor(x, max_len=None):
    if max_len is None:
        max_len = max([x[i].size(0) for i in range(len(x))])
    
    out_list = list()
    for i, batch in enumerate(x):
        if len(batch.shape) == 1:
            pad_size = (0, max_len - batch.size(0))
        else:
            pad_size = (0, 0, 0, max_len - batch.size(0))
            
        one_batch = F.pad(batch, pad_size)
        out_list.append(one_batch)
    
    out_padded = torch.stack(out_list)
    return out_padded