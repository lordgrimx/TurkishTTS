import torch
import torch.nn as nn

class FFTBlock(nn.Module):
    def __init__(
        self,
        d_model,
        conv1d_filter_size,
        conv1d_kernel_size,
        n_layer,
        n_head,
        d_k,
        d_v,
        dropout=0.1
    ):
        super(FFTBlock, self).__init__()
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.conv_layers = nn.ModuleList([
            ConvBlock(d_model, conv1d_filter_size, conv1d_kernel_size)
            for _ in range(n_layer)
        ])
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size-1)//2
        )
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v