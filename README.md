# mmExpert: Integrating Large Language Models for Comprehensive mmWave Data Synthesis and Understanding

<div align="center">

**A Multimodal Large Language Model for mmWave Data Understanding and Human Motion Analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2509.16521-red.svg)](https://arxiv.org/abs/2509.16521)

</div>

## 🌟 Overview

**mmExpert-LLM** is the Large Language Model component of the [mmExpert](https://arxiv.org/abs/2509.16521) project, an innovative mmWave understanding framework that integrates Large Language Models for comprehensive mmWave data synthesis and understanding. This project addresses the high costs associated with mmWave data acquisition and annotation by leveraging LLMs to automate the generation of synthetic mmWave radar datasets.

The system processes mmWave signals (representing human motion and activity data) and generates natural language descriptions, answers questions about motion patterns, and provides detailed analysis of human movement sequences. This enables zero-shot generalization in real-world environments and facilitates the successful deployment of large models for mmWave understanding.

> **📄 Paper**: [mmExpert: Integrating Large Language Models for Comprehensive mmWave Data Synthesis and Understanding](https://arxiv.org/abs/2509.16521)  
> **👥 Authors**: Yifan Yan, Shuai Yang, Xiuzhen Guo, Xiangguang Wang, Wei Chow, Yuanchao Shu, Shibo He (Zhejiang University)  
> **📅 Published**: ACM MobiHoc '25

## ✨ Key Features

### 🔄 **Multimodal Architecture**
- **mmWave Signal Processing**: Advanced CLIP-based encoder for mmWave radar data understanding
- **Language Model Integration**: Built on Microsoft Phi-3-mini-4k-instruct for natural language processing
- **Cross-Modal Alignment**: Seamless integration between mmWave signals and text representations

### 🎯 **Core Capabilities**
- **mmWave Data Understanding**: Process and interpret mmWave radar signals for human activity recognition
- **Synthetic Data Generation**: Leverage LLMs to generate synthetic mmWave datasets for specific scenarios
- **Zero-shot Generalization**: Train models capable of generalizing to real-world environments without extensive data collection
- **Human Motion Analysis**: Generate detailed descriptions and analysis of human movements from mmWave data

### 🚀 **Advanced Features**
- **Data Generation Flywheel**: Automated synthetic dataset creation using LLMs
- **LoRA Fine-tuning**: Parameter-efficient training with Low-Rank Adaptation
- **Flash Attention**: Optimized attention mechanisms for better performance
- **Multi-GPU Support**: Distributed training and inference capabilities
- **Comprehensive Evaluation**: Multiple metrics for mmWave understanding assessment

## 🏗️ Architecture

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended)

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd mmExpert

# Create conda environment (recommended)
conda create -n mmexpert python=3.8
conda activate mmexpert

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Required Dependencies

```bash
# Core dependencies
pip install transformers>=4.30.0
pip install torch>=2.0.0
pip install pytorch-lightning
pip install einops
pip install sentence-transformers
pip install timm
pip install peft
pip install swanlab

# Evaluation dependencies
pip install pycocoevalcap
pip install scikit-learn
pip install scipy
```

## 🚀 Quick Start

### 1. Data Preparation

Prepare your mmWave radar data in the following format:

```python
# mmWave data structure
{
    "filefolder": "path/to/mmwave/files",
    "fileindex": "mmwave_001",
    "captions": ["A person walking forward"],
    "questions": {
        "q1": {"question": "What is the person doing?", "answer": "Walking"}
    }
}
```

### 2. Training

#### Stage 1: CLIP Pre-training
```bash
# Train the wave encoder
python run_model.py --config config/clip_v1.11_bs64.yaml --version clip_v1.11_bs64
```

#### Stage 2: LLM Fine-tuning
```bash
# Fine-tune the multimodal model
bash scripts/train_r4a8.sh feature/clip_v1.11_bs64/
```

### 3. Inference

```python
from src.wavellm import WaveLLMForCausalLM
import torch

# Load the trained model
model = WaveLLMForCausalLM.from_pretrained("path/to/trained/model")
model.get_model().load_wave_encoder("path/to/encoder")

# Prepare input
mmwave_signal = torch.randn(1, 1, 496, 128)  # Example mmWave radar signal
question = "Describe the human motion shown in this mmWave signal."

# Generate response
with torch.no_grad():
    response = model.generate(
        input_wave_embeds=mmwave_signal,
        input_ids=tokenizer.encode(question, return_tensors="pt"),
        max_length=512
    )
```

## 📊 Dataset Support

### Supported Datasets

- **HumanML3D**: Large-scale human motion dataset for mmWave understanding
- **Custom mmWave Data**: Support for custom mmWave radar signals
- **Synthetic Data**: LLM-generated synthetic mmWave datasets for specific scenarios
- **Real-time Data**: Live mmWave radar data processing capabilities

### Data Format

The system supports mmWave radar data represented as wave signals with dimensions `[batch_size, channels, height, width]` where:
- `height`: Temporal dimension (e.g., 496 frames)
- `width`: Feature dimension (e.g., 128 features representing radar signal characteristics)

## 🔧 Configuration

### Model Configuration

Key configuration parameters in `config/clip_v1.11_bs64.yaml`:

```yaml
model_cfg:
  params:
    encoder_cfg:
      model_name: 'vit_base_patch16_224'
      image_resolution: [496, 128]
      embed_dim: 256
    text_cfg:
      model_name: 'sentence-transformers/paraphrase-MiniLM-L6-v2'
      embed_dim: 384
    context_length: 24
    transformer_width: 256
    transformer_heads: 8
    transformer_layers: 1
```

### Training Configuration

```yaml
# LoRA configuration
lora_r: 4
lora_alpha: 8
lora_dropout: 0.05
lora_bias: "none"

# Training parameters
learning_rate: 5e-4
per_device_train_batch_size: 6
num_train_epochs: 1
```

## 📈 Evaluation

### Available Metrics

- **BLEU**: Text generation quality for mmWave descriptions
- **METEOR**: Semantic similarity in motion understanding
- **ROUGE**: Text summarization quality for activity descriptions
- **CIDER**: Consensus-based evaluation for mmWave understanding
- **SPICE**: Semantic propositional evaluation
- **Semantic Similarity**: Cross-modal alignment between mmWave signals and text

### Sample Outputs

Here are some example outputs from the mmExpert-LLM model:

**Example 1: Posture Analysis**
- **Question**: "What is the intention of the person?"
- **Model Prediction**: "The person seems to be adjusting their posture or demonstrating a gesture."
- **Ground Truth**: "The person seems to be moving their arms casually, possibly adjusting posture."

**Example 2: Dance Recognition**
- **Question**: "What is the intention of the person according to the wave signal?"
- **Model Prediction**: "The person intends to dance the waltz."
- **Ground Truth**: "To perform a solo dance in a repeated square pattern."

### Running Evaluation

```bash
# Run comprehensive evaluation
python evaluation/LanguageEvaluator.py --model_path path/to/model --data_path path/to/test_data
```

## 🛠️ Advanced Usage

### Custom Training

```python
# Custom dataset integration
from src.datasets.wavellm_dataset import WaveCaptionDataset

# Create custom dataset
dataset = WaveCaptionDataset(
    data_root="path/to/your/data",
    split="train",
    tokenizer=tokenizer
)

# Custom training loop
from src.trainer.train_llm import train

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    learning_rate=5e-4,
    num_train_epochs=3,
    lora=True,
    lora_r=8,
    lora_alpha=16
)
```

### Model Customization

```python
# Custom wave encoder
from src.model.clip import ImageEncoder

custom_encoder = ImageEncoder(
    model_name='vit_base_patch16_224',
    embed_dim=512,
    image_resolution=[496, 128]
)

# Custom projection layers
projection_layers = nn.Linear(512, model.config.hidden_size)
```

## 📁 Project Structure

```
mmExpert/
├── src/
│   ├── wavellm/                 # Core WaveLLM implementation
│   │   ├── modeling_wavellm.py  # Main model architecture
│   │   └── conversation.py      # Conversation templates
│   ├── model/
│   │   ├── clip.py             # CLIP-based wave encoder
│   │   └── clip_loss.py        # Loss functions
│   ├── datasets/
│   │   ├── wavellm_dataset.py  # Dataset implementation
│   │   └── base_dataset.py     # Base dataset utilities
│   ├── trainer/
│   │   ├── train_llm.py        # Training script
│   │   └── eval_llm.py         # Evaluation script
│   └── misc/
│       └── tools.py            # Utility functions
├── config/                     # Configuration files
├── evaluation/                 # Evaluation scripts
├── scripts/                    # Training scripts
├── dataset/                    # Dataset storage
└── output/                     # Model outputs
```
