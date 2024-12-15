import torch
import torch.nn as nn
from collections import OrderedDict

class VariancePredictor(nn.Module):
    def __init__(self, encoder_dim):
        super(VariancePredictor, self).__init__()
        
        self.input_size = encoder_dim
        self.filter_size = 256
        self.kernel = 3
        self.conv_output_size = 256
        self.dropout = 0.5
        
        self.conv_layer = nn.Sequential(
            OrderedDict([
                ("conv1d_1", nn.Conv1d(
                    self.input_size,
                    self.filter_size,
                    self.kernel,
                    padding=(self.kernel-1)//2
                )),
                ("relu_1", nn.ReLU()),
                ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                ("dropout_1", nn.Dropout(self.dropout)),
                ("conv1d_2", nn.Conv1d(
                    self.filter_size,
                    self.filter_size,
                    self.kernel,
                    padding=1
                )),
                ("relu_2", nn.ReLU()),
                ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                ("dropout_2", nn.Dropout(self.dropout))
            ])
        )
        
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        return out


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VarianceAdaptor(nn.Module):
    def __init__(self, encoder_dim, n_mel_channels, max_seq_len):
        super(VarianceAdaptor, self).__init__()
        
        self.duration_predictor = VariancePredictor(encoder_dim)
        self.pitch_predictor = VariancePredictor(encoder_dim)
        self.energy_predictor = VariancePredictor(encoder_dim)
        
        self.length_regulator = LengthRegulator()
        
        self.pitch_embedding = nn.Conv1d(
            1,
            encoder_dim,
            kernel_size=3,
            padding=1
        )
        
        self.energy_embedding = nn.Conv1d(
            1,
            encoder_dim,
            kernel_size=3,
            padding=1
        )

    def get_pitch_embedding(self, x, target, prediction, control):
        if target is not None:
            embedding = self.pitch_embedding(target.unsqueeze(1))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(prediction.unsqueeze(1))
        return embedding.transpose(1, 2)

    def get_energy_embedding(self, x, target, prediction, control):
        if target is not None:
            embedding = self.energy_embedding(target.unsqueeze(1))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(prediction.unsqueeze(1))
        return embedding.transpose(1, 2)

    def forward(
        self,
        x,
        src_len,
        duration_target=None,
        pitch_target=None,
        energy_target=None,
        max_len=None,
        duration_control=1.0,
        pitch_control=1.0,
        energy_control=1.0
    ):
        # Duration tahminleri
        duration_prediction = self.duration_predictor(x)
        
        # Length regulation
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(
                torch.round(torch.exp(duration_prediction) - 1) * duration_control,
                min=0
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            
        # Pitch tahminleri
        pitch_prediction = self.pitch_predictor(x)
        pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, pitch_prediction, pitch_control)
        
        # Energy tahminleri
        energy_prediction = self.energy_predictor(x)
        energy_embedding = self.get_energy_embedding(
            x, energy_target, energy_prediction, energy_control)
            
        x = x + pitch_embedding + energy_embedding
        
        return (
            x,
            mel_len,
            duration_prediction,
            pitch_prediction,
            energy_prediction
        )


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = torch.nn.functional.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = torch.nn.functional.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded