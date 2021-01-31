from torch import nn

from ConSSL.utils.self_supervised import torchvision_ssl_encoder


class MLP(nn.Module):

    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):

    def __init__(self, encoder=None):
        super().__init__()

        if encoder is None:
            encoder = torchvision_ssl_encoder('resnet50')
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP()
        # Predictor
        self.predictor = MLP(input_dim=256)

    def forward(self, x):
        y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h
