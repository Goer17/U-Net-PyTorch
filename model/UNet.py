import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        num_channels = [3, 64, 128, 256, 512, 1024]
        
        self.encoder_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=num_channels[i], out_channels=num_channels[i + 1], kernel_size=3, padding=1),
            self.__conv_block(in_channels=num_channels[i + 1], out_channels=num_channels[i + 1])
        ) for i in range(5)])
        self.down = nn.MaxPool2d(kernel_size=2)
        
        self.ups = nn.ModuleList([nn.ConvTranspose2d(in_channels=num_channels[i], out_channels=num_channels[i - 1], kernel_size=2, stride=2) for i in range(5, 1, -1)])
        self.decoder_layers = nn.ModuleList([self.__conv_block(in_channels=num_channels[i], out_channels=num_channels[i - 1]) for i in range(5, 1, -1)])
        self.final_conv = self.__conv_block(in_channels=num_channels[1], out_channels=num_classes)
    
    def __conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.tensor):
        # Encoding Stage
        encoder_outputs = []
        for encoder_layer in self.encoder_layers[:-1]:
            x = encoder_layer(x)
            encoder_outputs.append(x)
            x = self.down(x)
        x = self.encoder_layers[-1](x)
        
        # Decoding Stage
        for i, out in enumerate(reversed(encoder_outputs)):
            x = self.ups[i](x)
            x = torch.cat([out, x], dim=1)
            x = self.decoder_layers[i](x)
        x = self.final_conv(x)
        
        return x


if __name__ == '__main__':
    # Test
    model = UNet(num_classes=10)
    print(f'Total parameters in the model: {sum(p.numel() for p in model.parameters())}')
    x = torch.randn(1, 3, 1024, 1024)
    y = model(x)
    
    print(y.shape)