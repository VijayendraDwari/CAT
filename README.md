# CAT (Cyclic Attention Transformer)

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

This repository contains the implementation of a Cyclic Attention Transformer (CAT), a novel architecture that leverages cyclic attention mechanisms to enhance contextual learning and performance in natural language processing tasks.

## 🌟 Highlights

- **Cyclic Attention Mechanism**: Introduces attention that includes cyclic shifts and gating for better contextual modeling
- **Non-Pretrained Model**: Designed as a non-pretrained transformer architecture, making it efficient for training from scratch
- **Benchmark Performance**: Achieves 91.00% accuracy on the AG News dataset with a vocabulary size of 50,002 and 120,000 training samples

## 🧩 Model Architecture

The Cyclic Attention Transformer incorporates:

- **Cyclic Attention Block**: Combines cyclic shifts and gating mechanisms to capture global dependencies
- **Hierarchical Multi-Head Attention**: Stacked layers of attention with intermediate normalization and feedforward networks
- **Global Pooling**: Adaptive average pooling for sequence aggregation before classification
- **Custom Tokenizer and Vocabulary**: Supports n-gram tokenization for enriched token representation
  
## Current Configuration
- embed_dim: Dimension of embeddings (default: 1024)
- num_heads: Number of attention heads (default: 8)
- ff_dim: Dimension of feedforward layers (default: 2048)
- num_layers: Number of transformer layers (default: 3)
- batch_size: Training batch size (default: 128)
- epochs: Number of training epochs (default: 5)

## 🔬 Methodology

- Tokenizer: A custom tokenizer supports both unigram and bigram tokenization.
- Vocabulary: Built from scratch with a maximum size of 50,002, including padding and unknown tokens.
- Attention Mechanisms:
        Cyclic shifts for attention terms.
        Gating to filter relevant attention outputs.
- Optimization: Trained with the AdamW optimizer and a cross-entropy loss function.

## 📊 Performance Metrics for AG News and DB Pedia Datasets

## Results on AG News dataset: 
The results obtained on the AG news dataset cosidering the fact that it's a non pretrained Transformer model seems to be acceptable.

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 91.00% |
| F1 Score  | 90.99% |
| Precision | 91.02% |
| Recall    | 91.00% |

### Results on DBpedia Benchmark

This repository also demonstrates the effectiveness of the Cyclic Attention Transformer architecture on the DBpedia benchmark dataset. Below are the results obtained:

#### Training Configuration

- **Dataset**: DBpedia 14
- **Training Samples**: 560,000
- **Testing Samples**: 70,000
- **Vocabulary Size**: 50,002
- **Batch Size**: 64
- **Epochs**: 5
- **Embedding Dimension**: 1024
- **Number of Heads**: 8
- **Feedforward Dimension**: 2048
- **Number of Layers**: 3
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5

#### Training and Evaluation Metrics

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 98.05% |
| F1 Score  | 98.05% |
| Precision | 98.06% |
| Recall    | 98.05% |

#### Training Loss Progression

| Epoch | Training Loss |
|-------|---------------|
| 1     | 0.1299        |
| 2     | 0.0681        |
| 3     | 0.0520        |
| 4     | 0.0416        |
| 5     | 0.0344        |

#### Key Observations

- The model achieves a high accuracy of 98.05% on the DBpedia test set.
- The Cyclic Attention Transformer showcases consistent training with a steadily decreasing loss over 5 epochs.
- These results further demonstrate the architecture's capacity for handling diverse, multi-class text classification tasks.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- PyTorch
- Transformers library

### Installation

1. Clone the repository:
```bash
git clone https://github.com/VijayendraDwari/CAT.git
cd CAT
```
📝 Citation

If you use this implementation in your research, please cite:

@misc{dwari2025cat,
  title={Cyclic Attention Transformer (CAT)},
  author={Vijayendra Dwari},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/VijayendraDwari/CAT}}
}


