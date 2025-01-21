# CAT (Cyclic Attention Transformer) 🔄

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

## 📖 Overview

The Cyclic Attention Transformer (CAT) is a novel transformer architecture that introduces cyclic attention mechanisms to enhance contextual learning. This non-pretrained model demonstrates exceptional performance on text classification tasks, achieving state-of-the-art results on multiple benchmarks without the need for extensive pretraining.

### Key Features

- 🔄 **Cyclic Attention Mechanism**: Advanced contextual modeling through cyclic shifts and gating
- 🚀 **Zero-Shot Architecture**: Efficient training from scratch without pretraining requirements
- 📊 **Strong Benchmark Results**: 
  - AG News Dataset: 91.00% accuracy
  - DBpedia Dataset: 98.05% accuracy
- 🎯 **Efficient Training**: Optimized for both speed and performance

## 🏗️ Architecture

### Core Components

1. **Cyclic Attention Block**
   - Innovative cyclic shift mechanism
   - Adaptive gating for attention filtering
   - Enhanced global dependency capture

2. **Multi-Head Attention System**
   - Hierarchical attention layers
   - Intermediate normalization
   - Advanced feedforward networks

3. **Processing Pipeline**
   - Custom n-gram tokenization
   - Global pooling for sequence aggregation
   - Specialized classification head

### Model Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| embed_dim | 1024 | Embedding dimension |
| num_heads | 8 | Number of attention heads |
| ff_dim | 2048 | Feedforward layer dimension |
| num_layers | 3 | Number of transformer layers |
| batch_size | 128 | Training batch size |
| epochs | 5 | Training epochs |

## 📊 Performance Analysis

### AG News Dataset Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 91.00% |
| F1 Score  | 90.99% |
| Precision | 91.02% |
| Recall    | 91.00% |

**Dataset Details:**
- Vocabulary Size: 50,002
- Training Samples: 120,000

### DBpedia Dataset Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.05% |
| F1 Score  | 98.05% |
| Precision | 98.06% |
| Recall    | 98.05% |

**Dataset Details:**
- Training Set: 560,000 samples
- Test Set: 70,000 samples
- Vocabulary Size: 50,002

#### Training Loss Progression

| Epoch | Loss   |
|-------|--------|
| 1     | 0.1299 |
| 2     | 0.0681 |
| 3     | 0.0520 |
| 4     | 0.0416 |
| 5     | 0.0344 |

## 🛠️ Implementation Details

### Technical Methodology

1. **Tokenization**
   - Custom tokenizer supporting unigram and bigram tokenization
   - Vocabulary size: 50,002 (including special tokens)

2. **Attention Implementation**
   - Cyclic shift attention mechanism
   - Gated attention filtering
   - Multi-head attention processing

3. **Training Configuration**
   - Optimizer: AdamW
   - Loss Function: Cross-entropy
   - Learning Rate: 5e-5

## 🚀 Getting Started

### Prerequisites

```bash
- Python 3.x
- PyTorch
- Transformers library
- Jupyter Notebook
```

I'll help you analyze and improve the README for your Cyclic Attention Transformer (CAT) implementation repository. Let me first check the current state of the repository and then propose improvements.

Let me get the current README to analyze it and then propose improvements.

Let me help you reorganize and enhance the README to make it more comprehensive and better structured. Here's a suggested improved version of the README:
Markdown

# CAT (Cyclic Attention Transformer) 🔄

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

## 📖 Overview

The Cyclic Attention Transformer (CAT) is a novel transformer architecture that introduces cyclic attention mechanisms to enhance contextual learning. This non-pretrained model demonstrates exceptional performance on text classification tasks, achieving state-of-the-art results on multiple benchmarks without the need for extensive pretraining.

### Key Features

- 🔄 **Cyclic Attention Mechanism**: Advanced contextual modeling through cyclic shifts and gating
- 🚀 **Zero-Shot Architecture**: Efficient training from scratch without pretraining requirements
- 📊 **Strong Benchmark Results**: 
  - AG News Dataset: 91.00% accuracy
  - DBpedia Dataset: 98.05% accuracy
- 🎯 **Efficient Training**: Optimized for both speed and performance

## 🏗️ Architecture

### Core Components

1. **Cyclic Attention Block**
   - Innovative cyclic shift mechanism
   - Adaptive gating for attention filtering
   - Enhanced global dependency capture

2. **Multi-Head Attention System**
   - Hierarchical attention layers
   - Intermediate normalization
   - Advanced feedforward networks

3. **Processing Pipeline**
   - Custom n-gram tokenization
   - Global pooling for sequence aggregation
   - Specialized classification head

### Model Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| embed_dim | 1024 | Embedding dimension |
| num_heads | 8 | Number of attention heads |
| ff_dim | 2048 | Feedforward layer dimension |
| num_layers | 3 | Number of transformer layers |
| batch_size | 128 | Training batch size |
| epochs | 5 | Training epochs |

## 📊 Performance Analysis

### AG News Dataset Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 91.00% |
| F1 Score  | 90.99% |
| Precision | 91.02% |
| Recall    | 91.00% |

**Dataset Details:**
- Vocabulary Size: 50,002
- Training Samples: 120,000

### DBpedia Dataset Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.05% |
| F1 Score  | 98.05% |
| Precision | 98.06% |
| Recall    | 98.05% |

**Dataset Details:**
- Training Set: 560,000 samples
- Test Set: 70,000 samples
- Vocabulary Size: 50,002

#### Training Loss Progression

| Epoch | Loss   |
|-------|--------|
| 1     | 0.1299 |
| 2     | 0.0681 |
| 3     | 0.0520 |
| 4     | 0.0416 |
| 5     | 0.0344 |

## 🛠️ Implementation Details

### Technical Methodology

1. **Tokenization**
   - Custom tokenizer supporting unigram and bigram tokenization
   - Vocabulary size: 50,002 (including special tokens)

2. **Attention Implementation**
   - Cyclic shift attention mechanism
   - Gated attention filtering
   - Multi-head attention processing

3. **Training Configuration**
   - Optimizer: AdamW
   - Loss Function: Cross-entropy
   - Learning Rate: 5e-5

## 🚀 Getting Started

### Prerequisites

```bash
- Python 3.x
- PyTorch
- Transformers library
- Jupyter Notebook
```
## Installation Steps

Clone the repository:
```bash
git clone https://github.com/VijayendraDwari/CAT.git
cd CAT
```
@misc{dwari2025cat,
  title={Cyclic Attention Transformer (CAT)},
  author={Vijayendra Dwari},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/VijayendraDwari/CAT}}
}

