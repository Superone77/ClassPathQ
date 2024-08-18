from datetime import datetime
from socket import gethostname
import json
from ..model import *

class config_json:
    def __init__(self, config_dict_dir):
        with open(config_dict_dir, 'r') as f:
            self.config_dict = json.load(f)
        self.network_config_dict = self.config_dict['network_config']
        self.dataset_config_dict = self.config_dict['dataset_config']
        self.training_config_dict = self.config_dict['training_config']
        self.refine_config_dict = self.config_dict['refine_config']
        self._init_dataset_parameters()
        self._init_network_parameters()
        self._init_training_parameters()
        self._init_refine_parameters()

    def _init_dataset_parameters(self):
        self.batch_size = self.dataset_config_dict['batch_size']
        self.dataset = self.dataset_config_dict['dataset']

    def _init_network_parameters(self):
        self.device = self.network_config_dict['device']

    def _init_training_parameters(self):
        self.training_lr = self.training_config_dict['learning_rate']
        self.training_momentum = self.training_config_dict['momentum']
        self.training_weight_decay = self.training_config_dict['weight_decay']
        self.training_epoch = self.training_config_dict['epoch']

    def _init_refine_parameters(self):
        self.refine_lr = self.refine_config_dict['learning_rate']
        self.refine_momentum = self.refine_config_dict['momentum']
        self.refine_weight_decay = self.refine_config_dict['weight_decay']
        self.refine_epoch = self.refine_config_dict['epoch']
    
    def print_config(self):
        print(self.network_config_dict)
        print(self.dataset_config_dict)
        print(self.training_config_dict)
        print(self.refine_config_dict)

    def load_net_KD(self):
        model = self.load_net(self.refine_config_dict['KD_name'])
        model.to(self.device)
        return model

    def load_net(self, net_kind=None):
        if net_kind == None:
            net_kind = self.network_config_dict['network']
        elif net_kind == 'vggsmall_c10_ref':
            from models.vggsmall_baseline import vggsmall
            model = vggsmall(class_num=10)
            return model
        elif net_kind == 'vggsmall_c100_ref':
            from models.vggsmall_baseline import vggsmall
            model = vggsmall(class_num=100)
            return model
        elif net_kind == 'resnet20_c10_ref':
            from models.resnet20_c10 import resnet20_c10
            model = resnet20_c10()
            model.set_quant(False, False)
            return model
        elif net_kind == 'resnet20_c10_1_ref':
            from models.resnet20_c10_1 import resnet20_c10_1
            model = resnet20_c10_1()
            model.set_quant(False, False)
            return model
        elif net_kind == 'resnet20_c100_ref':
            from models.resnet20_c100 import resnet20_c100
            model = resnet20_c100()
            model.set_quant(False, False)
            return model
        if net_kind == 'vggsmall_c10':
            from models.vggsmall_c10 import get_model
            model = get_model()
            return model
        elif net_kind == 'vggsmall_c100':
            from models.vggsmall_c100 import get_model
            model = get_model()
            return model
        elif net_kind == 'resnet20_c10':
            from models.resnet20_c10 import get_model
            model = get_model()
            return model
        elif net_kind == 'resnet20_c100':
            from models.resnet20_c100 import get_model
            model = get_model()
            return model
        elif net_kind == 'resnet20_c10_1':
            from models.resnet20_c10_1 import get_model
            model = get_model()
            return model
        else:
            raise Exception('No such network')

    def instance_name_generate(self):
        prefix = self.network_config_dict['prefix']
        date = datetime.today().strftime('%Y%m%d%H%M%S')
        host_name = gethostname()
        return prefix+date+host_name

    def get_total_dic(self):
        return self.config_dict