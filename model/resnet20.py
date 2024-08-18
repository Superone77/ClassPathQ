
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import ModelQuant
from ..classpathq.op_quant import Conv2dQuant,LinearQuant,ActivationQuant 

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dQuant(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.re1 = nn.ReLU()
        self.conv2 = Conv2dQuant(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.re2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.planes)
            )

    def forward(self, x):
        out = self.re1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.re2(out)
        return out

class ResNetExp1_quant(ModelQuant):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_quant, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.re1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.re1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def set_quant_config(self, config_dict):
        wquant_flag = config_dict['wquant_flag']
        iquant_flag = config_dict['iquant_flag']
        for i, layer in enumerate(self.modules()):
            if isinstance(layer, Conv2dQuant) or isinstance(layer, LinearQuant) or isinstance(layer, ActivationQuant):
                layer.wquant_flag = wquant_flag
                layer.iquant_flag = iquant_flag

    def set_bits_list(self, bits_list_all, act_bits = 8):
        firsttime = True
        bit_index = 0
        for i, layer in enumerate(self.modules()):
            # ignore the first layer
            if firsttime:
                if isinstance(layer, nn.Conv2d) and not isinstance(layer, Conv2dQuant):
                    firsttime = False
                    bit_index += 1
            if isinstance(layer, Conv2dQuant) or isinstance(layer, LinearQuant):
                layer.bits_list_w = bits_list_all[bit_index]
                layer.bits_list_i = len(bits_list_all[bit_index-1]) * [act_bits]
                bit_index += 1

    def get_average_weight(self,bit_widths_s):
        # 计算所有权重的均值
        conv_sum = 0
        conv_cnt = 0
        for i, sublist in enumerate(bit_widths_s[1:19]):
            conv_cnt += len(sublist) * 9 * len(bit_widths_s[i])
            for num in sublist:
                conv_sum += num * 9 * len(bit_widths_s[i])

        # 计算所有元素的均值
        current_mean_bit_width = conv_sum / conv_cnt
        # print(conv_cnt)
        return current_mean_bit_width
    
    




class ResNetExp5_quant(ModelQuant):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet_quant, self).__init__()
        self.expand = 5
        self.in_planes = 16 * self.expand

        self.conv1 = nn.Conv2d(3, 16*self.expand, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*self.expand)
        self.re1 = nn.ReLU()
        strides = [[1] * num_blocks[0], [2] + [1] * (num_blocks[1] - 1), [2] + [1] * (num_blocks[2] - 1)]
        self.layer1 = self._make_layer(block, 16*self.expand, num_blocks[0], strides[0])
        self.layer2 = self._make_layer(block, 32*self.expand, num_blocks[1], strides[1])
        self.layer3 = self._make_layer(block, 64*self.expand, num_blocks[2], strides[2])
        self.linear = nn.Linear(64*self.expand, num_classes)

    def _make_layer(self, block, planes, num_blocks, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.re1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def set_quant_config(self, config_dict):
        wquant_flag = config_dict['wquant_flag']
        iquant_flag = config_dict['iquant_flag']
        for i, layer in enumerate(self.modules()):
            if isinstance(layer, Conv2dQuant) or isinstance(layer, LinearQuant) or isinstance(layer, ActivationQuant):
                layer.wquant_flag = wquant_flag
                layer.iquant_flag = iquant_flag

    def set_bits_list(self, bits_list_all, act_bits = 8):
        firsttime = True
        bit_index = 0
        for i, layer in enumerate(self.modules()):
            # ignore the first layer
            if firsttime:
                if isinstance(layer, nn.Conv2d) and not isinstance(layer, Conv2dQuant):
                    firsttime = False
                    bit_index += 1
            if isinstance(layer, Conv2dQuant) or isinstance(layer, LinearQuant):
                layer.bits_list_w = bits_list_all[bit_index]
                layer.bits_list_i = len(bits_list_all[bit_index-1]) * [act_bits]
                bit_index += 1

    def get_average_weight(self):
        weight_num_total = 0
        quant_weight_num_total = 0

        for i, layer in enumerate(self.modules()):
            if isinstance(layer, Conv2dQuant):
                layer.get_quant_mac()
                weight_num_total += layer.weight_num_total
                quant_weight_num_total += layer.quant_weight_num_total

            elif isinstance(layer, LinearQuant):
                layer.get_quant_mac()
                weight_num_total += layer.weight_num_total
                quant_weight_num_total += layer.quant_weight_num_total

        return 1.0*quant_weight_num_total/weight_num_total

def resnet20_c10():
    #TODO: hardcode max of input and weight
    amax_i = torch.tensor([1.7141, 1.4851, 2.6795, 1.3347, 3.3723, 1.3308, 3.1975, 1.6874, 1.9181, 0.8388, 2.0955, 0.7935, 2.4038, 1.2915, 1.7105, 1.1081, 2.5874, 0.8812])
    amax_w = torch.tensor([0.4230, 0.3429, 0.2953, 0.2528, 0.3879, 0.2601, 0.2382, 0.3137, 0.2036, 0.1475, 0.2539, 0.1874, 0.1547, 0.1578, 0.1616, 0.1355, 0.1275, 0.1065])
    net = ResNetExp1_quant(BasicBlock, [3, 3, 3])
    amax_count = 0
    for module in net.modules():
        if isinstance(module, Conv2dQuant):
            module.amax_i = amax_i[amax_count]
            module.amax_w = amax_w[amax_count]
            amax_count += 1
    return net


def resnet20_c100():
    amax_i = torch.tensor([4.0999, 2.5832, 6.1280, 1.6425, 6.8898, 1.5541, 5.3414, 2.1204, 4.0964, 1.4713, 3.9610, 1.4545, 4.1016, 1.7253, 2.9278, 1.8998, 3.6537, 2.1468])
    amax_w = torch.tensor([0.5266, 0.4087, 0.4056, 0.3114, 0.3802, 0.2586, 0.2654, 0.2120, 0.1918, 0.1401, 0.2032, 0.1517, 0.1936, 0.1850, 0.1864, 0.1724, 0.1729, 0.1372])
    net = ResNetExp5_quant(BasicBlock, [3, 3, 3])
    amax_count = 0
    for module in net.modules():
        if isinstance(module, Conv2dQuant):
            module.amax_i = amax_i[amax_count]
            module.amax_w = amax_w[amax_count]
            amax_count += 1
    return net


def get_model_resnet20_c10():
    return resnet20_c10()

def get_model_resnet20_c100():
    return resnet20_c100()