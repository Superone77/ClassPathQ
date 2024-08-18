import torch.nn as nn
from ..core import quantizer_act

class ActivationQuant(OpQuant,nn.Module):
    def __init__(self, quant_flag = False, amax = None):
        nn.Module.super().__init__()
        OpQuant.super().__init__()

        self.bits_list = None
        self.quant_flag = quant_flag
        self.amax = amax

    def forward(self, x):
        if self.quant_flag == False:
            return x
        else:
            return quantizer_act(x, self.bits_list, self.amax)