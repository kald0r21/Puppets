# src/scripts/custom_policy.py

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Definicja sieci jest teraz w jednym, oddzielnym pliku
class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        h, w, c = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy_obs_hwc = th.zeros(1, h, w, c)
            dummy_obs_chw = dummy_obs_hwc.permute(0, 3, 1, 2)
            n_flat = self.cnn(dummy_obs_chw).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim), nn.ReLU(),
            nn.Dropout(0.1)
        )
        self._features_dim = features_dim

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs.permute(0, 3, 1, 2).float() / 255.0
        x = self.cnn(x)
        x = self.linear(x)
        return x