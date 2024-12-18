import torch
import torch.nn as nn

class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Loss ağırlıkları
        self.mel_weight = 1.0
        self.duration_weight = 0.1
        self.pitch_weight = 0.1
        self.energy_weight = 0.1

    def mel_loss(self, mel_predictions, mel_targets):
        """Mel spektrogram loss hesaplama"""
        return self.mae_loss(mel_predictions, mel_targets)

    def duration_loss(self, duration_predictions, duration_targets):
        """Duration loss hesaplama"""
        if duration_predictions is None or duration_targets is None:
            return torch.tensor(0.0).to(duration_targets.device if duration_targets is not None else 'cuda')
        return self.mse_loss(duration_predictions.float(), duration_targets.float())

    def pitch_loss(self, pitch_predictions, pitch_targets):
        """Pitch loss hesaplama"""
        if pitch_predictions is None or pitch_targets is None:
            return torch.tensor(0.0).to(pitch_targets.device if pitch_targets is not None else 'cuda')
        return self.mse_loss(pitch_predictions.float(), pitch_targets.float())

    def energy_loss(self, energy_predictions, energy_targets):
        """Energy loss hesaplama"""
        if energy_predictions is None or energy_targets is None:
            return torch.tensor(0.0).to(energy_targets.device if energy_targets is not None else 'cuda')
        return self.mse_loss(energy_predictions.float(), energy_targets.float())

    def forward(self, mel_predictions, mel_targets, duration_predictions=None, 
                duration_targets=None, pitch_predictions=None, pitch_targets=None,
                energy_predictions=None, energy_targets=None):
        
        mel_loss = self.mel_weight * self.mel_loss(mel_predictions, mel_targets)
        duration_loss = self.duration_weight * self.duration_loss(duration_predictions, duration_targets)
        pitch_loss = self.pitch_weight * self.pitch_loss(pitch_predictions, pitch_targets)
        energy_loss = self.energy_weight * self.energy_loss(energy_predictions, energy_targets)
        
        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss
        
        return total_loss