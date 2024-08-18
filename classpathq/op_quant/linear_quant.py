import torch.nn as nn
from ..core import quantizer_act
from .op_quant import OpQuant

class LinearQuant(OpQuant,nn.Linear):
    def __init__(self, in_features, out_features, bias=False, wquant_flag = False, iquant_flag = False, amax_w = None, amax_i = None):
        OpQuant.__init__()
        nn.Linear.super().__init__(in_features, out_features, bias)

        self.bits_list_w = None
        self.bits··_list_i = None
        self.bquantizer = None
        self.bias_flag = bias
        self.wquant_flag = wquant_flag
        self.iquant_flag = iquant_flag
        self.amax_w = amax_w
        self.amax_i = amax_i

        self.quant_mac = 0

    def forward(self, input):

        x = input if self.iquant_flag == False else quantizer_act(input, self.bits_list_i, 'linear', self.amax_i)
        weight = self.weight if self.wquant_flag == False else quantizer_weight(self.weight, self.bits_list_w, 'linear', self.amax_w)
        if self.bias_flag:
            bias, breturn_value = self.bias if self.bquantizer is None else self.bquantizer(self.bias)
        else:
            bias, breturn_value = None, None

        outputs = F.linear(x, weight, bias)

        return outputs

    def get_quant_mac(self):
        self.weight_num_total = self.in_features * self.out_features
        self.mac_total = self.weight_num_total
        self.quant_weight_num_total = 0
        for bit in self.bits_list_w:
            self.quant_weight_num_total += bit * self.in_features
        self.quant_mac_total = self.quant_weight_num_total