# Quick Start

## About RepGAN

This is a combine of [RepVgg](https://github.com/DingXiaoH/RepVGG) and [HiFiGAN](https://github.com/jik876/hifi-gan),
which can fuse the `MRF Block` in the HiFiGAN.

For example: HiFiGAN use `3, 7, 11` as the kernel sizes in the ResnetBlock, and using RepGAN, we only have one kernel
size `11`

## Training

u can use the official HiFiGAN and replace the model using this.

## Other tricks

u can use `ISTFT` or `MultiBand` to accelerate the model. 