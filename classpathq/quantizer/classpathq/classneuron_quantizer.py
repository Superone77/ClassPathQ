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



class ClassNeuronQuantizer(object):
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
        
        self.score_border_list = config_dict['quant_config']['score_border_list']
        self.score_bit_list = config_dict['quant_config']['score_bit_list']
        self.uniform_bit = config_dict['quant_config']['uniform_bit']
        self.search_border_list = config_dict['quant_config']['search_border_list']
        self.search_bit_list = config_dict['quant_config']['search_bit_list']
        self.step_length = config_dict['quant_config']['step_length']
        self.step_epoch = config_dict['quant_config']['step_epoch']
        self.protect_counter = config_dict['quant_config']['protect_counter']
        self.search_target = config_dict['quant_config']['search_target']
        self.pruning_decay = config_dict['quant_config']['pruning_decay']
        self.search_target_q = config_dict['quant_config']['search_target_q']
        self.act_quant = config_dict['quant_config']['act_quant']
        self.act_bit = config_dict['quant_config']['act_bit']
        

        self.total_epoch = self.training_epoch + self.refine_epoch

    

    def process_quant(self):
        self.init_KD()
        # epoch = self.training_epoch

        # torch.save(self.model.state_dict(), "pre_trained/resnet20x2_ori_model.ckpt")
        # print(total_scores)

        
        scorer = Scorer(self.model, self.class_num, self.device)
        total_scores = scorer.get_neuron_score(self.stat_loader, 10 ** (-50))
        print(self.score_border_list)
        if self.act_quant:
            self.model.set_quant(True, True)
        else:
            self.model.set_quant(True)
        bit_list_all = self.gen_bit_list(total_scores, self.score_border_list, self.score_bit_list,
                                            method='piecewise')
        # print(bit_list_all)
        self.model.set_bits_list(bit_list_all, self.act_bit)

        max_bit_list_all = self.score_to_bit_search(total_scores, self.search_target)

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

    # target: target of reference accuracy
    def score_to_bit_search(self, total_scores, target):
        bit_current = 1
        counter = 0
        acc, loss = self.test(self.valloader)
        print(acc, loss)
        while bit_current <= len(self.search_border_list):
            if bit_current > 1 and self.search_border_list[bit_current - 2] > self.search_border_list[bit_current - 1]:
                self.search_border_list[bit_current - 1] = self.search_border_list[bit_current - 2]
            bit_list_all = self.gen_bit_list(total_scores, self.search_border_list, self.search_bit_list,
                                             method='piecewise')
            # print(bit_list_all)
            for l in bit_list_all:
                print(len(l), end=' ')
            print('\n')
            if self.act_quant:
                self.model.set_quant(True, True)
            else:
                self.model.set_quant(True)
            self.model.set_bits_list(bit_list_all)

            acc, loss = self.test(self.trainloader)

            weight_num_quant = self.model.get_average_weight()
            if bit_current <= 1:
                org_target = target
                target = self.pruning_decay * target * (0.8 ** (bit_current - 1))
                if acc < target or weight_num_quant < self.search_target_q:
                    self.search_border_list[bit_current - 1] -= self.step_length
                    bit_current += 1
                    target = org_target
                    continue
            else:
                if weight_num_quant < self.search_target_q:
                    self.search_border_list[bit_current - 1] -= self.step_length
                    bit_current += 1
                    target = org_target
                    continue
            target = org_target

            self.search_border_list[bit_current - 1] += self.step_length
            if self.search_border_list[bit_current - 1] >= self.class_num:
                self.search_border_list[bit_current - 1] = self.class_num
                bit_current += 1
            # print(self.search_border_list)
            # print(self.model.get_average_weight())
            print('--------------------------------')
            counter += 1
        return bit_list_all

    def gen_bit_list(self, scores, score_border_list, score_bit_list, method='uniform', uniform_bits=8):
        bit_list_total = []
        for index, layer in enumerate(scores):
            if len(layer.shape) == 2:
                bit_list_total.append(
                    self._score_to_bit(layer[0], score_border_list, score_bit_list, method, uniform_bits))
            elif len(layer.shape) == 4:
                list_tmp = []
                for i in range(layer.shape[1]):
                    list_tmp.append(layer[:, i, :, :].max())
                if index == 0:
                    bit_list_total.append(
                        self._score_to_bit(list_tmp, score_border_list, score_bit_list, 'uniform', uniform_bits))
                else:
                    bit_list_total.append(
                        self._score_to_bit(list_tmp, score_border_list, score_bit_list, method, uniform_bits))
            else:
                raise ValueError("Wrong shape of scores")
        return bit_list_total

    def _score_to_bit(self, scores, score_border_list, score_bit_list, method='uniform', uniform_bits=8):
        bit_list = []
        if method == 'piecewise':
            for index, element in enumerate(scores):
                append_constraint = False
                for i, e in enumerate(score_border_list):
                    if element <= e:
                        bit_list.append(score_bit_list[i])
                        append_constraint = True
                        break
                if element > score_border_list[-1] and not append_constraint:
                    bit_list.append(score_bit_list[-1])
        elif method == 'uniform':
            for element in scores:
                bit_list.append(uniform_bits)

        return bit_list

    def _finetune_KD(self, start_epoch, end_epoch):
        kd_trainer = KD_trainer(self.model, self.device, self.trainloader, self.valloader, self.testloader, self.stat_loader, self.base_criterion,
                 self.instance_name, self.config_dict)
        for epoch in range(start_epoch, end_epoch):
            acc = kd_trainer.train(epoch, self.trainloader, self.optimizer_refine, self.base_criterion)
            self.scheduler_refine.step()
        return acc, epoch
    
    
