import torch
import torch.nn as nn


class ModelQuant(nn.Module):
    def __init__(self):
        super.__init__()
        pass

    def forward(self, inputs):
        pass

    def set_quant_config(self,config_dict = {}):
        pass
    
    def get_average_weight(self):
        return 0

    def get_average_bit(self,bit_widths_s):
        return 0