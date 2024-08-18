import copy
import torch
class BitGiver:
    def __init__(self, exp_cfg,bits_init,model, weight_target_bit, act_bit, testloader, device, base_criterion,model_name):
        self.model = model
        self.weight_target_bit = weight_target_bit
        self.act_bit = act_bit
        self.testloader = testloader
        self.device = device
        self.base_criterion = base_criterion
        self.model_name = model_name
        self._exp_cfg = exp_cfg
        self._bits_init = bits_init
        self._bits_list_all = _bits_init
        
    def set_bit_for_cls(temp_bit_widths, cls_path_location, cls_idx, cls_bits, bit, pass_layer_list):
        for act_idx, filter_idx in cls_path_location[cls_idx]:
            if act_idx in pass_layer_list:
                continue
            temp_bit_widths[cls_idx][act_idx][filter_idx] = bit
        return temp_bit_widths

    def update_bit_widths(temp_bit_widths, cls_path_location, pass_layer_list):
        bit_widths = copy.deepcopy(temp_bit_widths[0])  # Initialize with the first class bit widths
        for cls_idx in range(1, len(temp_bit_widths)):  # Start from the second class
            for act_idx, filter_idx in cls_path_location[cls_idx]:
                if act_idx in pass_layer_list:
                    continue
                bit_widths[act_idx][filter_idx] = max(bit_widths[act_idx][filter_idx],
                                                    temp_bit_widths[cls_idx][act_idx][filter_idx])
        return bit_widths

    def set_bit_for_path(bit_widths, cls_path_location, cls_idx, cls_bits, bit,pass_layer_list):
        num_classes = len(cls_bits)
        # In the main loop of bit_search function:
        temp_bit_widths = [copy.deepcopy(self._bits_init) for _ in
                        range(num_classes)]  # Reset temp_bit_widths to the latest bit_widths
        cls_bits[cls_idx] = bit
        for j in range(num_classes):
            self.set_bit_for_cls(temp_bit_widths, cls_path_location, j, cls_bits, cls_bits[j],pass_layer_list)
        bit_widths = self.update_bit_widths(temp_bit_widths, cls_path_location,pass_layer_list)
        return bit_widths
    
    def test(self, testloader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.base_criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            loss_output = test_loss / (batch_idx + 1)
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss_output, acc, correct, total))

        return acc, loss
    
    def bit_init(self, bits_list_all):
        self._bits_list_all = bits_list_all
    
    def cls_bit_init(self, sorted_classes, desired_bit_width, init_type='NBit',
                     advise_sorted_cls_bit=[1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3]):
        if desired_bit_width < 3:
            N = 4
        else:
            N = 6
        if init_type == 'NBit':
            cls_bits = [N] * len(sorted_classes)
        elif init_type == 'Manual':
            cls_bit = [N] * len(sorted_classes)
            if len(advise_sorted_cls_bit) != len(cls_bit):
                raise Exception("False Manual Advised Sorted Cls Bit")

            for idx, sorted_idx in enumerate(sorted_classes):
                cls_bit[sorted_idx] = advise_sorted_cls_bit[idx]
        else:
            raise Exception('NO SUCH CLS_BIT INIT TYPE')
        return cls_bits
    
    def pruning(self,bit_widths,important_neuron_locations,cls_bits,sorted_classes,search_target_q):
        print("pruning")
        if search_target_q < 3:
            N = 3
        elif search_target_q < 4:
            N = 4
        else:
            N = 5
        cls_idx = sorted_classes[-1]
        bit = N
        act_bit = self.act_bit
        bit_widths = self.set_bit_for_path(bit_widths, important_neuron_locations,
                                      cls_idx, cls_bits, bit, self.model_name)
        self.model.set_bits_list(bit_widths, act_bit)
        current_mean_bit_width = self.model.get_average_weight()
        current_accuracy, _ = self.test(self.testloader)
        print('pruning:','cls_idx:', sorted_classes[-1], ',n:', bit, ',cls_bits:', cls_bits,',accuracy:', current_accuracy,
              ',bit:', current_mean_bit_width)
        idx = 0
        while current_mean_bit_width > search_target_q:
            cls_idx = sorted_classes[idx]
            if cls_bits[cls_idx] == 0:
                idx += 1
                continue
            bit_widths = self.set_bit_for_path(bit_widths, important_neuron_locations,
                                          cls_idx, cls_bits, 0, self.model_name)
            cls_bits[cls_idx] -= 1
            self.model.set_bits_list(bit_widths, act_bit)
            current_mean_bit_width = self.model.get_average_weight()
            current_accuracy, _ = self.test(self.testloader)
            print('pruning:','cls_idx:', cls_idx, ',n:', 0, ',cls_bits:', cls_bits, ',accuracy:', current_accuracy,
                  ',bit:', current_mean_bit_width)
        return cls_bits, current_mean_bit_width, bit_widths

    def bit_search(self, important_neuron_locations, sorted_classes,T1, R,reverse_flag = True,init_type='NBit',advise_sorted_cls_bit=[1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3],pruning_flag=False):
        bits_list_all = self._bits_list_all
        print('T1:', T1, 'R:', R,'reverse_flag:',reverse_flag)
        search_target_q = self.weight_target_bit
        act_bit = self.act_bit
        desired_bit_width = search_target_q
        search_target_q_int = int(search_target_q)
        if desired_bit_width < 3:
            N = 4
        else:
            N = 6
        act_bit = act_bit
        bit_widths = bits_list_all
        cls_bits = self.cls_bit_init(sorted_classes, desired_bit_width,init_type,advise_sorted_cls_bit)
        baseline = 1 / (len(sorted_classes) - 1)
        self.model.set_bits_list(bit_widths, act_bit)
        current_accuracy, _ = self.test(self.testloader)
        target_accuracy = T1 * (current_accuracy - baseline) + baseline
        n_min = 0
        last_iter = False
        ori_bit_widths = copy.deepcopy(bit_widths)
        ori_cls_bits = copy.deepcopy(cls_bits)
        for i in range(len(sorted_classes)):
            bit = None
            cls_idx = sorted_classes[i]
            for n in range(N, n_min, -1):
                bit = n
                ori_bit_widths = copy.deepcopy(bit_widths)
                ori_cls_bits = copy.deepcopy(cls_bits)
                bit_widths = set_bit_for_path(bit_widths, important_neuron_locations,
                                              cls_idx, cls_bits, bit, self.model_name)
                cls_bits[cls_idx] = bit
                self.model.set_bits_list(bit_widths, act_bit)
                current_accuracy, _ = self.test(self.testloader)
                current_mean_bit_width = self.model.get_average_weight()
                print('search:','cls_idx:', sorted_classes[i], ',n:', n, ',cls_bits:', cls_bits, ',accuracy:', current_accuracy,
                      ',bit:', current_mean_bit_width)
                if i == len(sorted_classes) - 1:
                    last_iter = True
                if current_mean_bit_width < desired_bit_width:
                    if current_mean_bit_width < search_target_q_int and pruning_flag:
                        cls_bits, current_mean_bit_width, bit_widths = self.pruning(bit_widths,
                                                                                    important_neuron_locations,
                                                                                    cls_bits,
                                                                                    sorted_classes,
                                                                                    search_target_q)
                    return cls_bits, current_mean_bit_width, bit_widths
                if current_accuracy < target_accuracy:
                    bit_widths = copy.deepcopy(ori_bit_widths)
                    cls_bits = copy.deepcopy(ori_cls_bits)
                    target_accuracy = (target_accuracy - baseline) * R + baseline
                    break
        if last_iter and current_mean_bit_width > desired_bit_width:
            # 重新搜索时，优先降低之前结果中比特数较大的类的比特数
            while True:
                cls_bits_sorted = []
                # print(sorted_classes)
                if reverse_flag:
                    for n in range(N, n_min, -1):
                        # print(n)
                        for i in sorted_classes:
                            if cls_bits[i] == n:
                                print(i)
                                cls_bits_sorted.append(i)
                else:
                    for n in range(n_min+2, N+1):
                        for i in sorted_classes:
                            if cls_bits[i] == n:
                                cls_bits_sorted.append(i)
                # print(cls_bits)
                # print(cls_bits_sorted)
                cls_idx = cls_bits_sorted[0]
                cls_bits[cls_idx] -= 1
                if cls_bits[cls_idx] <= 0:
                    cls_bits[cls_idx] = 1
                bit_widths = set_bit_for_path(bit_widths, important_neuron_locations, cls_idx, cls_bits,
                                              cls_bits[cls_idx], self.model_name)
                self.model.set_bits_list(bit_widths, act_bit)
                current_accuracy, _ = self.test(self.testloader)
                current_mean_bit_width = self.model.get_average_weight()
                print('second_step:','cls_idx:', cls_idx, ',n:', cls_bits[cls_idx], 'cls_bits:', cls_bits, ',accuracy:',
                      current_accuracy, ',bit:', current_mean_bit_width)
                if current_mean_bit_width <= desired_bit_width:
                    if current_mean_bit_width < search_target_q_int and pruning_flag:
                        cls_bits, current_mean_bit_width, bit_widths = self.pruning(bit_widths,
                                                                                    important_neuron_locations,
                                                                                    cls_bits,
                                                                                    sorted_classes,
                                                                                    search_target_q)
                    return cls_bits, current_mean_bit_width, bit_widths

        if current_mean_bit_width < search_target_q_int and pruning_flag:
            cls_bits, current_mean_bit_width, bit_widths = self.pruning(bit_widths,
                                                                        important_neuron_locations,
                                                                        cls_bits,
                                                                        sorted_classes,
                                                                        search_target_q)
        return cls_bits, current_mean_bit_width, bit_widths