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

## 📊 Performance Metrics

Results on AG News dataset:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 91.00% |
| F1 Score  | 90.99% |
| Precision | 91.02% |
| Recall    | 91.00% |

## 📂 Dataset

The implementation was evaluated on the AG News dataset:

- **Training Samples**: 120,000
- **Classes**: 4 (World, Sports, Business, Sci/Tech)
- **Dataset Split**: Standard AG News test set

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


