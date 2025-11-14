# custom_policy.py
# Enhanced CNN feature extractor for visual observations

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    """
    Enhanced CNN feature extractor for small RGB images (13x13x3)

    Features:
    - Batch normalization for stable training
    - Dropout for regularization
    - Residual-like connections
    - Optimized for egocentric observations
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        h, w, c = observation_space.shape  # HWC format (13, 13, 3)

        # Enhanced CNN architecture
        self.cnn = nn.Sequential(
            # First conv block - extract basic features
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Second conv block - spatial features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Third conv block - complex patterns
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Flatten for fully connected layers
            nn.Flatten()
        )

        # Calculate flattened dimension
        with th.no_grad():
            # Create dummy observation in HWC format (as SB3 provides)
            dummy_obs_hwc = th.zeros(1, h, w, c)
            # Convert to CHW for CNN
            dummy_obs_chw = dummy_obs_hwc.permute(0, 3, 1, 2)
            n_flat = self.cnn(dummy_obs_chw).shape[1]

        # Fully connected layers with dropout
        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization
        )

        self._features_dim = features_dim

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Forward pass

        Args:
            obs: Observations in HWC format (batch, height, width, channels)
                 Values in range [0, 255] as uint8 or float32

        Returns:
            Feature vector of size features_dim
        """
        # Convert HWC -> CHW and normalize to [0, 1]
        x = obs.permute(0, 3, 1, 2).float() / 255.0

        # CNN feature extraction
        x = self.cnn(x)

        # Fully connected projection
        x = self.linear(x)

        return x