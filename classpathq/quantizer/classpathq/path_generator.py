import numpy as np
import torch
import copy

class PathGenerator:
    def __init__(self, model, score, cls_score,device,model_name):
        self.model = model
        self.score = score
        self.cls_score = cls_score
        self.new_class_filter_rank = [{} for _ in range(len(cls_score))]
        self.device = device
        self.model_name = model_name

    def get_last_layer_num(self):
        if 'vgg' in self.model_name:
            return 8
        elif 'resnet20_c10' in self.model_name:
            return 19
        else:
            raise Exception('no such model name')


    def gen_critical_pathway(self, cls_max_importance_per_filter, number_class, sparsity=0.5,
                             shared_threshold_cross_cls=False, score_type='jaccard_similarity',loc_for_unimportance = None):
        # print(cls_max_importance_per_filter)
        # 拷贝class_filter_rank，但不包括每个类的key为0和8的部分
        # 创建新的数据结构，用于去除第0层和第19层
        last_layer_num = self.get_last_layer_num()
        if loc_for_unimportance is None:
            loc_for_unimportance = int(number_class/2)
        new_class_filter_rank = self.new_class_filter_rank
        print("sparsity:",sparsity,"shared_threshold_cross_cls:",shared_threshold_cross_cls, "score_type:",score_type)
        if 'vgg' in self.model_name:
            for cls_idx in range(number_class):
                for activation_idx, filter_scores in cls_max_importance_per_filter[cls_idx].items():
                    if activation_idx != 0 and activation_idx != 8:
                        new_class_filter_rank[cls_idx][activation_idx] = copy.deepcopy(filter_scores)
        elif 'resnet20' in self.model_name:
            for cls_idx in range(number_class):
                for activation_idx, filter_scores in cls_max_importance_per_filter[cls_idx].items():
                    if activation_idx != 0:
                        new_class_filter_rank[cls_idx][activation_idx] = copy.deepcopy(filter_scores)
        else:
            raise Exception('no such model name')


        thresholds = [0] * number_class
        total = 0
        total_importance = 0
        # all_conv_scores = []
        total = 0
        total_importance = 0
        total = 0
        if shared_threshold_cross_cls:

            all_conv_scores = []
            for cls_idx in range(number_class):
                # 计算每个类的阈值
                # all_conv_scores = []
                for activation_idx, filter_scores in new_class_filter_rank[cls_idx].items():
                    all_conv_scores.extend(filter_scores.view(-1).tolist())
                num_filters_to_preserve = int((1 - sparsity) * len(all_conv_scores))
                total += len(all_conv_scores)
                total_importance += num_filters_to_preserve
                sorted_filters = sorted(all_conv_scores)
                threshold = sorted_filters[-num_filters_to_preserve]
                thresholds[cls_idx] = threshold

            thresholds = [thresholds[9]] * number_class
        else:

            for cls_idx in range(number_class):
                # 计算每个类的阈值
                all_conv_scores = []
                for activation_idx, filter_scores in new_class_filter_rank[cls_idx].items():
                    all_conv_scores.extend(filter_scores.view(-1).tolist())
                num_filters_to_preserve = int((1 - sparsity) * len(all_conv_scores))
                total += len(all_conv_scores)
                total_importance += num_filters_to_preserve
                sorted_filters = sorted(all_conv_scores)
                threshold = sorted_filters[-num_filters_to_preserve]
                thresholds[cls_idx] = threshold

        print(total_importance)
        print(total)
        print(total_importance / total)
        # 输出阈值
        print("thresholds:", thresholds)

        # 创建相同形状的数据结构存储重要filter
        important_filters = copy.deepcopy(new_class_filter_rank)

        # 创建一个记录重要filter位置的列表，最后一个类是记录所有不重要的神经元的位置
        important_neuron_locations = []

        min_one_important = copy.deepcopy(new_class_filter_rank[0])

        # 初始化min_one_important
        for activation_idx, filter_scores in new_class_filter_rank[0].items():
            for i, scores in enumerate(filter_scores):
                min_one_important[activation_idx][i] = 0

        # 遍历每个类别
        for cls_idx in range(number_class):
            # print(cls_idx)
            threshold = thresholds[cls_idx]
            # 遍历得分
            cls_important_neuron_locations = []
            for activation_idx, filter_scores in new_class_filter_rank[cls_idx].items():
                for i, scores in enumerate(filter_scores):
                    max_score = max(scores.view(-1))
                    # for j, score in enumerate(scores.view(-1)):
                    # x, y = j % scores.size(1), j // scores.size(1)
                    if max_score >= threshold:
                        important_filters[cls_idx][activation_idx][i] = 1
                        cls_important_neuron_locations.append((activation_idx, i))  # 0 = conv
                        min_one_important[activation_idx][i] = 1
                    else:
                        # important_conv_filters[cls_idx][activation_idx][i][y][x] = 0
                        important_filters[cls_idx][activation_idx][i] = 0

            important_neuron_locations.append(cls_important_neuron_locations)
        print(important_neuron_locations[1] == important_neuron_locations[2])
        cls_important_neuron_locations = []
        # 找到不重要的
        for activation_idx, filter_scores in new_class_filter_rank[0].items():
            for i, scores in enumerate(filter_scores):
                if min_one_important[activation_idx][i] == 0:
                    cls_important_neuron_locations.append((activation_idx, i))

        important_neuron_locations.append(cls_important_neuron_locations)

        critical_pathways = []

        for cls_idx in range(number_class):
            critical_neurons = set()

            # Add important convolutional neurons
            for activation_idx, filters in important_filters[cls_idx].items():
                for i, value in enumerate(filters):
                    if value == 1:
                        critical_neurons.add((activation_idx, i))

            critical_pathways.append(critical_neurons)

        num_classes = number_class

        similarity_matrix = self.scoring_classpath(critical_pathways, num_classes, score_type)

        print("Jaccard Similarity Matrix:\n", similarity_matrix)

        # 升序排列，与别的类重合程度越高，排名越靠前，第一个为不重要的神经元
        sorted_classes = sorted(range(num_classes), key=lambda i: sum(similarity_matrix[i]))
        imp_sorted_classes = [number_class]
        sorted_classes.extend(imp_sorted_classes)
        temp = sorted_classes[number_class]
        sorted_classes[num_classes] = sorted_classes[loc_for_unimportance]
        sorted_classes[loc_for_unimportance] = temp
        print("Sorted Classes:", sorted_classes)

        return critical_pathways, important_neuron_locations, sorted_classes

    def scoring_classpath(self, critical_pathways, num_classes, score_type='jaccard_similarity'):
        similarity_matrix = np.zeros((num_classes, num_classes))
        if score_type == 'jaccard_similarity':
            for i in range(num_classes):
                for j in range(num_classes):
                    similarity_matrix[i][j] = self._jaccard_similarity(critical_pathways[i], critical_pathways[j])
            return similarity_matrix

        elif score_type == 'interference_degree':
            for i in range(num_classes):
                for j in range(num_classes):
                    similarity_matrix[i][j] = self.interference_degree(critical_pathways, i, j)
            return similarity_matrix
        else:
            raise Exception('NO SUCH CLASSPATH SCORE TYPE')

    # caculate the interference_degree from source path to target path, by sum up all the score of neuron in target path
    # which is also the important neuron of source path
    def interference_degree(self, critical_pathways, cls_idx_source, cls_idx_target):
        new_class_filter_rank = self.new_class_filter_rank
        interference_degree = torch.zeros(1).to(self.device)

        for activation_idx, filter_scores in new_class_filter_rank[cls_idx_target].items():
            for i, scores in enumerate(filter_scores):
                if (activation_idx, i) in critical_pathways[cls_idx_source]:
                    max_score = max(scores.view(-1))
                    interference_degree += max_score
        return interference_degree

    def _jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union