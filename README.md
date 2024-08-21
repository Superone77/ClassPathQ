# ClassPathQ: A Neural Network Quantization based on Classification Difficulty

## Abstract
In recent years, the increasing complexity and size of neural network models have led to significant challenges in computational demands, storage, and deployment, especially in resource-constrained environments. This thesis proposes a novel quantization strategy for neural networks, termed class-path-based quantization, which considers the classification difficulty of specific classes to optimize the allocation of bit-widths in quantized models. By identifying and prioritizing critical paths—sets of neurons and filters crucial for class-specific outputs—the proposed method allocates higher precision to more influential paths, maintaining model accuracy while reducing computational overhead.

The approach involves scoring the importance of each neuron and filter for individual classes and constructing critical paths based on these scores. Sensitivity scores are then calculated to measure the impact of quantization on overall model performance. A search algorithm is introduced to determine the optimal bit-width configuration, which is subsequently refined through knowledge distillation to restore any lost accuracy.

Experimental results demonstrate that class-path-based quantization outperforms traditional methods, particularly in complex classification tasks. This approach allows for effective model compression and acceleration, making advanced neural network models more accessible and efficient across various hardware platforms.

## A Scalable Quantization Framework
To explore the relationship between class scoring, class paths, and quantization through extensive experimentation, we developed a reusable and scalable per-channel quantization framework. This framework abstracts the processes of scoring, path extraction, quantization, and other configurations into independent, loosely coupled components. Consequently, it allows flexible adjustment of each component via scripts, enabling the generation of a vast array of experimental results.

As a modular quantization framework, it empowers developers to customize various aspects of neural network quantization. The key features include:

* Construction of models using the PyTorch library with model-level quantization metrics.
* Integration of operators specifically designed for network quantization, with defined operator-level quantization metrics.
* Development of custom trainers that seamlessly integrate training and fine-tuning into the model workflow.
* A unified configuration interface for network quantization and a consistent trainer interface for both operators and models, simplifying the definition and integration of quantizers.
* Support for per-channel quantization across all currently implemented operators.

