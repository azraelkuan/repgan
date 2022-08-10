# Quick Start

## About RepGAN

This is a combine of [RepVgg](https://github.com/DingXiaoH/RepVGG) and [HiFiGAN](https://github.com/jik876/hifi-gan),
which can fuse the `MRF Block` in the HiFiGAN. U can check the samples in `samples`

> Important, it can speed up HiFiGAN_v1 about 2x times using one cpu, 
> and with other tricks like `MultiBand`, `ISTFTNet`, `IDWT`, u can speed up to 4x times. 

For example: HiFiGAN use `3, 7, 11` as the kernel sizes in the ResnetBlock, and using RepGAN, we only have one kernel
size `11`

## Train

u can use the official [HiFiGAN](https://github.com/jik876/hifi-gan) or [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) and replace the model using this.