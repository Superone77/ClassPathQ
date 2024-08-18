import sys

import torch
import torch.nn as nn
import torch.optim as optim

from ..classpathq.configration import ConfigJson
from ..classpathq.dataloader import prepare_datasets
from ..classpathq.quantizer import ClassPathQuantizer

import numpy as np
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def main(argv):
    seed = random.randint(1, 100)
    setup_seed(seed)
    print(seed)
    config_dir = './configs/resnet20_c10_1_a2w2.json'
    config = ConfigJsons(config_dir)
    torch.cuda.set_device(config.device)
    device = torch.device(config.device)
    trainloader, valloader, testloader = prepare_datasets(config.dataset, config.batch_size)
    _, stat_loader, _ = prepare_datasets(config.dataset, 1, 0, True)

    instance_name = config.instance_name_generate()

    model = config.load_net()
    model.to(device)
    model_KD = config.load_net_KD()
    model_KD.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_training = optim.SGD(model.parameters(), lr=config.training_lr,
                                   momentum=config.training_momentum, weight_decay=config.training_weight_decay)
    scheduler_training = torch.optim.lr_scheduler.MultiStepLR(optimizer_training, milestones=[100, 150, 300], gamma=0.1)

    optimizer_refine = optim.SGD(model.parameters(), lr=config.refine_lr,
                                 momentum=config.refine_momentum, weight_decay=config.refine_weight_decay)
    scheduler_refine = torch.optim.lr_scheduler.MultiStepLR(optimizer_refine, milestones=[50,100,150,300], gamma=0.1)

    proc = ClassPathQuantizer(model, model_KD, device, trainloader, valloader, testloader, stat_loader, criterion,
                            optimizer_training, scheduler_training, optimizer_refine, scheduler_refine, instance_name,
                            config.get_total_dic())

    proc.process_quant()


if __name__ == "__main__":
    main(sys.argv[1:])
