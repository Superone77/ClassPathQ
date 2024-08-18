# ClassPathQ: A Neural Network Quantization based on Class Difficulty

## Abstract
In recent years, the increasing complexity and size of neural network models have led to significant challenges in computational demands, storage, and deployment, especially in resource-constrained environments. This thesis proposes a novel quantization strategy for neural networks, termed class-path-based quantization, which considers the classification difficulty of specific classes to optimize the allocation of bit-widths in quantized models. By identifying and prioritizing critical paths—sets of neurons and filters crucial for class-specific outputs—the proposed method allocates higher precision to more influential paths, maintaining model accuracy while reducing computational overhead.

The approach involves scoring the importance of each neuron and filter for individual classes and constructing critical paths based on these scores. Sensitivity scores are then calculated to measure the impact of quantization on overall model performance. A search algorithm is introduced to determine the optimal bit-width configuration, which is subsequently refined through knowledge distillation to restore any lost accuracy.

Experimental results demonstrate that class-path-based quantization outperforms traditional methods, particularly in complex classification tasks. This approach allows for effective model compression and acceleration, making advanced neural network models more accessible and efficient across various hardware platforms.



