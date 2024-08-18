import torch
import torch.nn as nn


class TaylorScorer:
    def __init__(self, model, input, device, label=None, output_orig=None):
        self.model = model
        self.input = input
        self.model.eval()
        self.device = device
        self.gradients = []
        self.neurons = []
        if label and output_orig is not None:
            self.label = label
            self.output_orig = output_orig
        else:
            self.output_orig = self.model(input).detach()
            self.label = self.output_orig.data.max(1)[1].item()
        self.handles_list = []
        self._hook_layers()

    def _hook_layers(self):
        def backward_hook_relu(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].to(self.device))

        def forward_hook_relu(module, input, output):
            self.neurons.append(output.to(self.device))
            return output

        for i, module in enumerate(self.model.modules()):
            if isinstance(module, nn.ReLU):
                self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                self.handles_list.append(module.register_full_backward_hook(backward_hook_relu))

    def remove_handles(self):
        for handle in self.handles_list:
            handle.remove()
        self.handles_list.clear()
        self.neurons = []
        self.gradients = []

    def _forward(self, input):
        self.neurons = []
        self.gradients = []
        self.model.zero_grad()
        output = self.model(input)
        return output

    def _number_of_elements(self):
        total = 0
        for layer in self.neurons:
            num_neurons_in_layer = layer.numel()
            total += num_neurons_in_layer
        return total

    def _compute_taylor_scores(self):
        first_order_taylor_scores = []
        self.gradients.reverse()
        for i, layer in enumerate(self.neurons):
            first_order_taylor_scores.append(torch.abs(torch.mul(layer, self.gradients[i])))

        return first_order_taylor_scores

    def get_taylor_scores(self, target=None, debug=False):
        initial_output = self._forward(self.input)
        initial_output = torch.nn.functional.softmax(initial_output, dim=1)
        initial_predicted_logit = initial_output.data.max(1)[0].item()
        initial_predicted_class = initial_output.data.max(1)[1].item()
        if debug:
            print("Initial output = {}".format(initial_predicted_logit))
            print('Initial predicted class {}: '.format(initial_predicted_class))
        label = torch \
            .tensor([self.label]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        initial_loss = criterion(initial_output, label)

        num_total = self._number_of_elements()
        if debug:
            print('initial loss {}'.format(initial_loss))
            print("total number of neurons: {}".format(num_total))

        if target == None:
            target = self.label

        output = self._forward(self.input)
        output[0, target].backward(retain_graph=True)
        first_order_taylor_scores = self._compute_taylor_scores()

        return first_order_taylor_scores


class Scorer:
    def __init__(self, model, class_num, device):
        self.model = model
        self.class_num = class_num
        self.device = device

    def get_scores(self, data, model, target, device, score_type='IS'):
        if score_type == 'IS':
            model.eval()
            scorer = TaylorScorer(model, data, device)
            scorse = scorer.get_taylor_scores(target=target)
            scorer.remove_handles()
            return scorse
        elif score_type == 'SS':
            pass
        else:
            raise Exception('No such score type')

    def get_statistic_IS(self, dataloader, threshold, max_num_per_class=100):

        self.model.eval()
        for target_index in range(self.class_num):
            dataiter = iter(dataloader)
            target_index_counter = 0
            for chosen in range(len(dataiter)):
                data, target = next(dataiter)
                if target[0].item() != target_index:
                    continue
                target_index_counter += 1
                data = data.to(self.device)
                output = self.model(data.clone())
                output = torch.nn.functional.softmax(output.detach(), dim=1)
                scores = self.get_scores(data.clone(), self.model, target, self.device)
                if target_index_counter == 1:
                    class_scores = [torch.zeros_like(scores[i]) for i in range(len(scores))]
                else:
                    for index, layer in enumerate(scores):
                        class_scores[index] += layer > threshold
                if target_index_counter >= max_num_per_class:
                    break
            if target_index == 0:
                total_scores = [torch.zeros_like(scores[i]) for i in range(len(scores))]
            for index, layer in enumerate(class_scores):
                total_scores[index] += layer / target_index_counter
        return total_scores

    def get_neuron_score(self, dataloader, threshold, score_type='IS', max_num_per_class=100):
        if score_type == 'IS':
            return self.get_statistic_IS(dataloader, threshold)
        elif score_type == 'SS':
            return self.get_statisric_SS(dataloader)
        else:
            raise Exception('NO SUCH SCORE TYPE')

    def get_neuron_score_for_path(self, dataloader, threshold, max_num_per_class=100):
        self.model.eval()
        class_scores_list = []  # 创建一个新列表来存储每个类别的评分
        for target_index in range(self.class_num):
            dataiter = iter(dataloader)
            target_index_counter = 0
            for chosen in range(len(dataiter)):
                data, target = next(dataiter)
                if target[0].item() != target_index:
                    continue
                target_index_counter += 1
                data = data.to(self.device)
                output = self.model(data.clone())
                output = torch.nn.functional.softmax(output.detach(), dim=1)
                scores = self.get_scores(data.clone(), self.model, target, self.device)
                if target_index_counter == 1:
                    class_scores = [torch.zeros_like(scores[i]) for i in range(len(scores))]
                else:
                    for index, layer in enumerate(scores):
                        class_scores[index] += layer > threshold
                if target_index_counter >= max_num_per_class:
                    break
            # 在添加到列表之前，先对 class_scores 进行归一化
            normalized_class_scores = [score / target_index_counter for score in class_scores]
            class_scores_list.append(normalized_class_scores)  # 将归一化的 class_scores 添加到列表中
            if target_index == 0:
                total_scores = [torch.zeros_like(scores[i]) for i in range(len(scores))]
            for index, layer in enumerate(class_scores):
                total_scores[index] += layer / target_index_counter

        max_importance_per_filter = {}

        for i, layer in enumerate(total_scores):
            if len(layer.shape) == 4:
                tmp = torch.empty(layer.shape[1])
                for j in range(layer.shape[1]):
                    tmp[j] = layer[:, j, :, :].max()
                total_scores[i] = tmp
                max_importance_per_filter[i] = total_scores[i]
            elif len(layer.shape) == 2:
                max_importance_per_filter[i] = total_scores[i][0]

        cls_max_importance_per_filter = []

        for cls_idx, class_scores in enumerate(class_scores_list):
            single_cls_max_importance_per_filter = {}
            for i, layer in enumerate(class_scores_list[cls_idx]):
                if len(layer.shape) == 4:
                    tmp = torch.empty(layer.shape[1])
                    for j in range(layer.shape[1]):
                        tmp[j] = layer[:, j, :, :].max()
                    class_scores_list[cls_idx][i] = tmp
                    single_cls_max_importance_per_filter[i] = class_scores_list[cls_idx][i]
                elif len(layer.shape) == 2:
                    single_cls_max_importance_per_filter[i] = class_scores_list[cls_idx][i][0]

            cls_max_importance_per_filter.append(single_cls_max_importance_per_filter)

        return max_importance_per_filter, cls_max_importance_per_filter  # 返回总分和每个类别的评分列表
