# CAT
 Implementation of a Cyclic Attention Transformer (CAT)
Cyclic Attention Transformer (CAT)

This repository contains the implementation of a Cyclic Attention Transformer (CAT), a novel architecture that leverages cyclic attention mechanisms to enhance contextual learning and performance on text classification tasks. The model has been tested and benchmarked on the AG News dataset, achieving competitive results.
🌟 Highlights

    Cyclic Attention Mechanism: Introduces attention that includes cyclic shifts and gating for better contextual modeling.
    Non-Pretrained Model: Designed as a non-pretrained transformer architecture, making it efficient for training from scratch.
    Benchmark Performance: Achieves 91.00% accuracy on the AG News dataset with a vocabulary size of 50,002 and 120,000 training samples.

🧩 Model Architecture

The Cyclic Attention Transformer incorporates:

    Cyclic Attention Block: Combines cyclic shifts and gating mechanisms to capture global dependencies.
    Hierarchical Multi-Head Attention: Stacked layers of attention with intermediate normalization and feedforward networks.
    Global Pooling: Adaptive average pooling for sequence aggregation before classification.
    Custom Tokenizer and Vocabulary: Supports n-gram tokenization for enriched token representation.

🚀 Results on AG News
Metric	Value
Accuracy	91.00%
F1 Score	90.99%
Precision	91.02%
Recall	91.00%
📂 Dataset

The AG News dataset was used for training and evaluation:

    Training Samples: 120,000
    Testing Samples: Full test set from the dataset
    Classes: 4 (World, Sports, Business, Sci/Tech)
