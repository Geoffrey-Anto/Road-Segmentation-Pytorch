from torch import nn


class Model(nn.Module):
    def __init__(self, in_channels=3, num_blocks=6, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(True),

            nn.Conv2d(features[0], features[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(True),

            nn.Conv2d(features[1], features[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(True),

            nn.Conv2d(features[2], features[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(True),
        )

        self.layers = []

        for i in range(num_blocks):
            self.layers.append(
                nn.Conv2d(features[3], features[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(features[3], features[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),

            nn.ConvTranspose2d(features[2], features[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),

            nn.ConvTranspose2d(features[1], features[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),

            nn.ConvTranspose2d(features[0], in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down(x)
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.up(x)
        return x


# def main():
#     m = Model(3, 5)
#     x = torch.randn((1, 3, 256, 256), requires_grad=True);
#     pred = m(x)
#     print(pred.shape)
#     print(m)
#     print(x.backward)
#
#
# if __name__ == "__main__":
#     main()
