import torch
import torch.nn as nn

from rep_conv import RepConv


class Upsample(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

        self.transposed_conv = nn.ConvTranspose1d(in_channels,
                                                  out_channels,
                                                  kernel_size=scale * 2,
                                                  stride=scale,
                                                  padding=scale // 2 + scale % 2,
                                                  output_padding=scale % 2)

    def forward(self, x):
        return self.transposed_conv(x)

    def inference(self, x):
        return self.forward(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 channels: int = 512,
                 kernel_sizes: tuple = (3, 7, 11),
                 dilations: tuple = (1, 3, 5),
                 use_additional_convs: bool = True):
        super(ResidualBlock, self).__init__()

        self.use_additional_convs = use_additional_convs

        self.act = nn.LeakyReLU(0.1)
        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(RepConv(channels, kernel_sizes, dilation=dilation))
            if use_additional_convs:
                self.convs2 += [RepConv(channels, kernel_sizes, dilation=1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            x = self.act(x)
            x = self.convs1[idx](x)
            if self.use_additional_convs:
                x = self.act(x)
                x = self.convs2[idx](x)
        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            x = self.act(x)
            x = self.convs1[idx].inference(x)
            if self.use_additional_convs:
                x = self.act(x)
                x = self.convs2[idx].inference(x)
        return x


class RepGANGenerator(nn.Module):

    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 channels=512,
                 kernel_size=7,
                 dropout=0.1,
                 upsample_scales=(8, 8, 2, 2),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 use_additional_convs=True,
                 use_weight_norm=True,
                 ):
        super(RepGANGenerator, self).__init__()

        # check hyper parameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(resblock_dilations) == len(upsample_scales)

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()

        self.non_linear = nn.LeakyReLU(0.1)
        for i in range(len(upsample_scales)):
            self.upsamples += [
                Upsample(
                    in_channels=channels // (2 ** i),
                    out_channels=channels // (2 ** (i + 1)),
                    scale=upsample_scales[i]
                )
            ]

            self.blocks += [
                ResidualBlock(
                    channels=channels // (2 ** (i + 1)),
                    kernel_sizes=resblock_kernel_sizes,
                    dilations=resblock_dilations[i],
                    use_additional_convs=use_additional_convs
                )
            ]

        self.output_conv = nn.Sequential(
            nn.Conv1d(channels // (2 ** (i + 1)), out_channels, kernel_size, padding=kernel_size // 2),
            nn.Tanh()
        )

        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x):
        x = self.input_conv(x)

        for i in range(len(self.upsamples)):
            x = self.upsamples[i](x)
            if self.training:
                x = self.blocks[i](x)
            else:
                x = self.blocks[i].inference(x)
            x = self.non_linear(x)
        x = self.output_conv(x)
        return x

    def convert_weight_bias(self):
        def _convert_weight_bias(m):
            if isinstance(m, RepConv):
                m.convert_weight_bias()

        self.apply(_convert_weight_bias)

    def apply_weight_norm(self):
        """Apply weight normalization module from all the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                nn.utils.weight_norm(m)
            if isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m, dim=1)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):

        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)


if __name__ == '__main__':
    voc = RepGANGenerator(use_weight_norm=False)
    dummy = torch.rand(1, 80, 100)
    y1 = voc(dummy)

    params = 0
    for n, p in voc.named_parameters():
        params += p.numel()
    print('model size: {}M'.format(params / 1e6))

    voc.convert_weight_bias()
    voc.eval()
    y2 = voc(dummy)

    print((y1 - y2).mean())
