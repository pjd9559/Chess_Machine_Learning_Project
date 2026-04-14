import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)


class ChessNet(nn.Module):
    def __init__(self, in_channels: int = 18, channels: int = 128, n_blocks: int = 6, n_moves: int = 4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        self.p_conv = nn.Conv2d(channels, 32, 1)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 8 * 8, n_moves)

        self.v_conv = nn.Conv2d(channels, 32, 1)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.body(x)

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.flatten(1)
        policy_logits = self.p_fc(p)

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v))

        return policy_logits, value
