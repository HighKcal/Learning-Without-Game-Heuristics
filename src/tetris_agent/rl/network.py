"""The fixed 235 -> 256 -> 256 -> 800 MLP baseline."""
from torch import nn


class QNetwork(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(235, 256), nn.ReLU(), nn.Linear(256, 256),
                         nn.ReLU(), nn.Linear(256, 800))
