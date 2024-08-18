from .quantizer import Quantizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import datetime
import os

from .bit_giver import BitGiver
from .path_generator import PathGenerator
from .scorer import Scorer
from ..KD_trainer import *



class ClassPathQuantizer(object):
    """
    basic class of quantizer
    """
    def __init__(self, model, model_KD, device, trainloader, valloader, testloader, stat_loader, criterion,
                 optimizer_training, scheduler_training, optimizer_refine, scheduler_refine, instance_name,
                 config_dict):
        self.model = model
        self.model.eval()
        self.model_KD = model_KD
        self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.stat_loader = stat_loader
        self.base_criterion = criterion
        self.optimizer_training = optimizer_training
        self.scheduler_training = scheduler_training
        self.optimizer_refine = optimizer_refine
        self.scheduler_refine = scheduler_refine
        self.load_config(config_dict)
        self.instance_name = instance_name

        self.gradients = []
        self.activations = []
        self.activations_mask = None
        self.weights_mask = None
        self.positive_activations_mask = []
        self.negative_activations_mask = []
        self.positive_weights_mask = []
        self.negative_weights_mask = []

        self.handles_list = []
        self.split_point = None

        self.weights_ori = []
        self.pruning_weights_mask = []

    def load_config(self, config_dict):
        # network_config
        self.netname = config_dict['network_config']['network']
        # dataset_config
        self.class_num = config_dict['dataset_config']['class_num']
        # training_config
        self.training_epoch = config_dict['training_config']['epoch']
        # refine_config
        self.learning_rate_refine = config_dict['refine_config']['learning_rate']
        self.refine_epoch = config_dict['refine_config']['epoch']
        self.KD_path = config_dict['refine_config']['KD_path']
        self.KD_name = config_dict['refine_config']['KD_name']
        self.KD_loss_alpha = config_dict['refine_config']['KD_loss_alpha']
        # quant_config
        self.classpath_flag = config_dict['quant_config']['classpath_flag']
        self.uniform_bit = config_dict['quant_config']['uniform_bit']
        self.search_target_q = config_dict['quant_config']['search_target_q']
        self.act_quant = config_dict['quant_config']['act_quant']
        self.act_bit = config_dict['quant_config']['act_bit']
        self.T1 = config_dict['quant_config']['T1']
        self.R = config_dict['quant_config']['R']
        self.sparsity = config_dict['quant_config']['sparsity']
        self.shared_threshold_cross_cls = config_dict['quant_config']['shared_threshold_cross_cls']
        self.score_type_for_path = config_dict['quant_config']['score_type_for_path']
        self.greedy_reverse_flag = config_dict['quant_config']['greedy_reverse_flag']
        self.cls_bit_init_type = config_dict['quant_config']['cls_bit_init_type']
        self.advise_sorted_cls_bit = config_dict['quant_config']['advise_sorted_cls_bit']
        self.loc_for_unimportance = config_dict['quant_config']['loc_for_unimportance']
        self.pruning_flag = config_dict['quant_config']['pruning_flag']

        self.total_epoch = self.training_epoch + self.refine_epoch

    

    def process_quant(self):
        self.init_KD()
        # epoch = self.training_epoch

        # torch.save(self.model.state_dict(), "pre_trained/resnet20x2_ori_model.ckpt")
        # print(total_scores)
        print("Score for Neuron Start")
        scorer = Scorer(self.model, self.class_num, self.device)
        total_score, class_scores = scorer.get_neuron_score_for_path(self.stat_loader, 10 ** (-50))
        print("Path Generation Start")
        sparsity_list = [0.3, 0.5, 0.6, 0.9]
        loc_for_unimportance_list = [int(self.class_num/2),
                                        int(self.class_num/2),
                                        int(self.class_num/2),
                                        self.class_num]
        max_cls_bits = []
        max_current_mean_bit_width = -1
        max_bit_list_all = []
        for i in range(len(sparsity_list)):
            sparsity = sparsity_list[i]
            loc_for_unimportance = loc_for_unimportance_list[i]
            path_generator = PathGenerator(self.model, total_score, class_scores, self.device, self.netname)
            critical_pathways, important_neuron_locations, sorted_classes = path_generator.gen_critical_pathway(
                class_scores, self.class_num, sparsity, self.shared_threshold_cross_cls,
                self.score_type_for_path,
                loc_for_unimportance)

            if self.act_quant:
                self.model.set_quant(True, True)
            else:
                self.model.set_quant(True)
            bit_giver = BitGiver(self.model, self.search_target_q, self.act_bit, self.testloader, self.device,
                                    self.base_criterion, self.netname)
            print("Bit Search Start")
            bits_list_all = bit_giver.bit_init()

            self.model.set_bits_list(bits_list_all, self.act_bit)

            cls_bits, current_mean_bit_width, bit_list_all = bit_giver.bit_search(important_neuron_locations,
                                                                                    sorted_classes, bits_list_all,
                                                                                    self.T1,
                                                                                    self.R,
                                                                                    self.greedy_reverse_flag,
                                                                                    self.cls_bit_init_type,
                                                                                    self.advise_sorted_cls_bit,
                                                                                    self.pruning_flag)
            print("sparsity:",sparsity,",loc_for_unimportance:",loc_for_unimportance,',current_mean_bit_width:',current_mean_bit_width)
            if current_mean_bit_width > max_current_mean_bit_width:
                max_current_mean_bit_width = current_mean_bit_width
                max_cls_bits = copy.deepcopy(cls_bits)
                max_bit_list_all = copy.deepcopy(bit_list_all)

        print("result of bit search:")
        print(max_current_mean_bit_width)
        print(max_cls_bits)
        print(max_bit_list_all)



        aver_weight = self.model.get_average_weight()
        print(aver_weight)
        self.model.set_bits_list(max_bit_list_all, self.act_bit)



        print("KD_loss_alpha:", self.KD_loss_alpha)
        acc, epoch = self._finetune_KD(epoch, epoch + self.refine_epoch)

        acc, _ = self.test(self.testloader)
        print(self.model.get_average_weight())

        return acc, bit_list_all

    def _finetune_KD(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch):
            acc = self.train_KD(epoch, self.trainloader, self.optimizer_refine, self.base_criterion)
            self.scheduler_refine.step()
        return acc, epoch
