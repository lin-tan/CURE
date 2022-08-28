import torch
import torch.nn as nn
from torch.nn.modules.utils import _single


class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, remove_future=False):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.remove_future = remove_future

        self.weight = torch.nn.Parameter(torch.Tensor(self.kernel_size[0], in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, x):
        output = torch.conv_tbc(x.contiguous(), self.weight, self.bias, self.padding[0])
        if self.remove_future:
            if self.kernel_size[0] > 1 and self.padding[0] > 0:
                # remove future timesteps added by padding
                output = output[:-self.padding[0], :, :]
        return output
