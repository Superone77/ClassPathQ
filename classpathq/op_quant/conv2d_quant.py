import torch.nn as nn
from ..core import quantizer_act, quantizer_weight
from .op_quant import OpQuant

class Conv2dQuant(OpQuant,nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, wquant_flag = False, iquant_flag = False, amax_w = None, amax_i = None):
        nn.Conv2d.super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        OpQuant.super().__init__()

        self.bits_list_w = None
        self.bits_list_i = None
        self.wquant_flag = wquant_flag
        self.iquant_flag = iquant_flag
        self.amax_w = amax_w
        self.amax_i = amax_i

        self.quant_mac = 0

        self.first_time = True

    def forward(self, input):
        if self.first_time:
            _, _, self.num_sliding = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, padding=self.padding).shape
            self.first_time = False

        x = input if self.iquant_flag == False else quantizer_act(input, self.bits_list_i, 'Conv2d', self.amax_i)
        weight = self.weight if self.wquant_flag == False else quantizer_weight(self.weight, self.bits_list_w, 'Conv2d', self.amax_w)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def get_quant_mac(self):
        ksize = self.in_channels*self.kernel_size[0]*self.kernel_size[1]
        self.weight_num_total = self.out_channels * ksize
        self.mac_total = self.weight_num_total * self.num_sliding
        self.quant_weight_num_total = 0
        for bit in self.bits_list_w:
            self.quant_weight_num_total += bit * ksize
        self.quant_mac_total = self.quant_weight_num_total * self.num_sliding