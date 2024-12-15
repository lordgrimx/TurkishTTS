import torch
import torch.nn as nn

class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel_predictions, mel_targets, duration_predictions=None, 
                duration_targets=None, pitch_predictions=None, pitch_targets=None,
                energy_predictions=None, energy_targets=None):
        
        # Mel-spectrogram loss
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        total_loss = mel_loss

        # Duration predictor loss
        if duration_predictions is not None and duration_targets is not None:
            duration_loss = self.mse_loss(duration_predictions, duration_targets.float())
            total_loss += duration_loss

        # Pitch predictor loss
        if pitch_predictions is not None and pitch_targets is not None:
            pitch_loss = self.mse_loss(pitch_predictions, pitch_targets.float())
            total_loss += pitch_loss

        # Energy predictor loss
        if energy_predictions is not None and energy_targets is not None:
            energy_loss = self.mse_loss(energy_predictions, energy_targets.float())
            total_loss += energy_loss

        return total_loss