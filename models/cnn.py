from torch import nn

class myNet(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        layers = []
        for in_channels, out_channels in ((3, 32), (32, 32), (32, 64)):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            )
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.extend([
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])

        layers.extend([
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = myNet()
    print(model)
