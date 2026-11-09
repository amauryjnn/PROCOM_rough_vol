from torch import nn


class CNN(nn.Module):
    """Calibrateur G_w : surface (54) -> paramètres (6)."""

    def __init__(self, n_maturities, n_moneyness, padding_mode="replicate", n_params=6):
        super().__init__()
        self.n_maturities = n_maturities
        self.n_moneyness = n_moneyness

        conv_args = dict(kernel_size=3, padding=1, padding_mode=padding_mode)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, **conv_args),
            nn.ELU(),
            nn.Conv2d(16, 32, **conv_args),
            nn.ELU(),
            nn.Conv2d(32, 32, **conv_args),
            nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * n_maturities * n_moneyness, 128),
            nn.ELU(),
            nn.Linear(128, n_params),
        )

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_maturities, self.n_moneyness)
        return self.head(self.features(x))
