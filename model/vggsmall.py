from .model import ModelQuant
from ..classpathq.op_quant import Conv2dQuant,LinearQuant,ActivationQuant 
import torch.nn.functional as F


class vggsmall_c10(ModelQuant):
    def __init__(self, num_classes = 10):
        super.__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True, momentum=0.1),
            nn.ReLU(),
            Conv2dQuant(128, 128, 3, padding=1, bias=False, amax_w = torch.tensor(0.0986), amax_i=torch.tensor(1.0435)),
            nn.BatchNorm2d(128, affine=True, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            Conv2dQuant(128, 256, 3, padding=1, bias=False, amax_w = torch.tensor(0.0582), amax_i=torch.tensor(0.9233)),
            nn.BatchNorm2d(256, affine=True, momentum=0.1),
            nn.ReLU(),
            Conv2dQuant(256, 256, 3, padding=1, bias=False, amax_w = torch.tensor(0.0518), amax_i=torch.tensor(0.4913)),
            nn.BatchNorm2d(256, affine=True, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            Conv2dQuant(256, 512, 3, padding=1, bias=False, amax_w = torch.tensor(0.0462), amax_i=torch.tensor(0.5951)),
            nn.BatchNorm2d(512, affine=True, momentum=0.1),
            nn.ReLU(),
            Conv2dQuant(512, 512, 3, padding=1, bias=False, amax_w = torch.tensor(0.0382), amax_i=torch.tensor(0.3991)),
            nn.BatchNorm2d(512, affine=True, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            LinearQuant(512*4*4, 1024, bias=False, amax_w = torch.tensor(0.0222), amax_i=torch.tensor(0.4653)),
            torch.nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            torch.nn.ReLU(),
            LinearQuant(1024, 1024, bias=False, amax_w = torch.tensor(0.0158), amax_i=torch.tensor(0.3431)),
            torch.nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            torch.nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(-1, 512*4*4)
        x = self.fc(x)
        return x

    def set_quant_config(self, config_dict):
        wquant_flag = config_dict['wquant_flag']
        iquant_flag = config_dict['iquant_flag']
        for i, layer in enumerate(self.modules()):
            if isinstance(layer, Conv2dQuant) or isinstance(layer, LinearQuant) or isinstance(layer, ActivationQuant):
                layer.wquant_flag = wquant_flag
                layer.iquant_flag = iquant_flag

    def set_bits_list(self, bits_list_all, act_bits = 8):
        firsttime = True
        hotfix = True
        bit_index = 0
        for i, layer in enumerate(self.modules()):
            # ignore the first layer
            if firsttime:
                if isinstance(layer, nn.Conv2d) and not isinstance(layer, Conv2dQuant):
                    firsttime = False
                    bit_index += 1
            if isinstance(layer, Conv2dQuant) or isinstance(layer, LinearQuant):
                layer.bits_list_w = bits_list_all[bit_index]
                # hotfix for the first linear layer
                if len(bits_list_all[bit_index-2]) == 512 and hotfix:
                    layer.bits_list_i = 8192 * [act_bits]
                    hotfix = False
                else:
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
    
    def get_average_bit(self,bit_widths_s):
        # 计算所有权重的均值
        conv_sum = 0
        conv_cnt = 0
        for i, sublist in enumerate(bit_widths_s[1:6]):
            conv_cnt += len(sublist) * 9 * len(bit_widths_s[i])
            for num in sublist:
                conv_sum += num * 9 * len(bit_widths_s[i])

        linear_sum = 0
        linear_cnt = 0
        # 两个全连接层手动加入
        linear_cnt += len(bit_widths_s[6]) * 512 * 4 * 4
        linear_cnt += 1024 * 1024

        for num in bit_widths_s[6]:
            linear_sum += num * 512 * 4 * 4
        for num in bit_widths_s[7]:
            linear_sum += num * 1024

        # 计算所有元素的均值
        current_mean_bit_width = (conv_sum + linear_sum) / (conv_cnt + linear_cnt)
        return current_mean_bit_width




class vggsmall_c100(ModelQuant):
    def __init__(self, num_classes = 100):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True, momentum=0.1),
            nn.ReLU(),
            Conv2dQuant(128, 128, 3, padding=1, bias=False, amax_w = torch.tensor(0.1198), amax_i=torch.tensor(1.1397)),
            nn.BatchNorm2d(128, affine=True, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            Conv2dQuant(128, 256, 3, padding=1, bias=False, amax_w = torch.tensor(0.0785), amax_i=torch.tensor(1.1575)),
            nn.BatchNorm2d(256, affine=True, momentum=0.1),
            nn.ReLU(),
            Conv2dQuant(256, 256, 3, padding=1, bias=False, amax_w = torch.tensor(0.0621), amax_i=torch.tensor(0.7427)),
            nn.BatchNorm2d(256, affine=True, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            Conv2dQuant(256, 512, 3, padding=1, bias=False, amax_w = torch.tensor(0.0489), amax_i=torch.tensor(0.5137)),
            nn.BatchNorm2d(512, affine=True, momentum=0.1),
            nn.ReLU(),
            Conv2dQuant(512, 512, 3, padding=1, bias=False, amax_w = torch.tensor(0.0356), amax_i=torch.tensor(0.4463)),
            nn.BatchNorm2d(512, affine=True, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            LinearQuant(512*4*4, 1024, bias=False, amax_w = torch.tensor(0.0429), amax_i=torch.tensor(0.4286)),
            torch.nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            torch.nn.ReLU(),
            LinearQuant(1024, 1024, bias=False, amax_w = torch.tensor(0.2902), amax_i=torch.tensor(2.4610)),
            torch.nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            torch.nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(-1, 512*4*4)
        x = self.fc(x)
        return x

    def set_quant_config(self, config_dict):
        wquant_flag = config_dict['wquant_flag']
        iquant_flag = config_dict['iquant_flag']
        for i, layer in enumerate(self.modules()):
            if isinstance(layer, Conv2dQuant) or isinstance(layer, LinearQuant) or isinstance(layer, ActivationQuant):
                layer.wquant_flag = wquant_flag
                layer.iquant_flag = iquant_flag


    def set_bits_list(self, bits_list_all, act_bits = 8):
        firsttime = True
        hotfix = True
        bit_index = 0
        for i, layer in enumerate(self.modules()):
            # ignore the first layer
            if firsttime:
                if isinstance(layer, nn.Conv2d) and not isinstance(layer, Conv2dQuant):
                    firsttime = False
                    bit_index += 1
            if isinstance(layer, Conv2dQuant) or isinstance(layer, LinearQuant):
                layer.bits_list_w = bits_list_all[bit_index]
                # hotfix for the first linear layer
                if len(bits_list_all[bit_index-2]) == 512 and hotfix:
                    layer.bits_list_i = 8192 * [act_bits]
                    hotfix = False
                else:
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

    def get_average_bit(self,bit_widths_s):
        # 计算所有权重的均值
        conv_sum = 0
        conv_cnt = 0
        for i, sublist in enumerate(bit_widths_s[1:6]):
            conv_cnt += len(sublist) * 9 * len(bit_widths_s[i])
            for num in sublist:
                conv_sum += num * 9 * len(bit_widths_s[i])

        linear_sum = 0
        linear_cnt = 0
        # 两个全连接层手动加入
        linear_cnt += len(bit_widths_s[6]) * 512 * 4 * 4
        linear_cnt += 1024 * 1024

        for num in bit_widths_s[6]:
            linear_sum += num * 512 * 4 * 4
        for num in bit_widths_s[7]:
            linear_sum += num * 1024

        # 计算所有元素的均值
        current_mean_bit_width = (conv_sum + linear_sum) / (conv_cnt + linear_cnt)
        return current_mean_bit_width


def get_model_vggsmall_c10():
    return vggsmall_c10()

def get_model_vggsmall_c100():
    return vggsmall_c100()

