# mmExpert

<div align="center">

**Integrating Large Language Models for Comprehensive mmWave Data Synthesis and Understanding**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2509.16521-red.svg)](https://arxiv.org/abs/2509.16521)

</div>

## 🌟 Overview

<div align="center">
  <img src="assets/overview.png" alt="mmExpert Architecture Overview" width="800">
</div>

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

The mmExpert system employs a sophisticated two-stage training architecture:

### Stage 1: CLIP Pre-training
- **Radar Encoder**: Vision Transformer (ViT) with adaptive patch sizing for multi-view mmWave signals
- **Text Encoder**: Pre-trained language models (BERT, RoBERTa, MiniLM, MPNet) with configurable freezing strategies
- **Cross-Modal Alignment**: Contrastive learning for radar-text alignment using CLIP/SigLIP loss
- **Multi-View Support**: Range-time, Doppler-time, and Azimuth-time spectrum processing

### Stage 2: LLM Fine-tuning
- **Base Model**: Microsoft Phi-3-mini-4k-instruct as the core language model
- **Parameter-Efficient Training**: LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
- **Wave Token Integration**: Specialized tokens for radar signal representation
- **Conversation Templates**: Structured input/output for mmWave understanding tasks

### Key Components
- **Core Abstractions**: Modular design with registry-based component management
- **Experiment Management**: Comprehensive configuration system with automated parameter printing
- **Distributed Training**: Multi-GPU support with DDP and FSDP strategies
- **Evaluation Framework**: Multiple metrics for mmWave understanding assessment

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

# Computer vision and visualization
pip install opencv-python

# Evaluation dependencies
pip install pycocoevalcap
pip install scikit-learn
pip install scipy

# Configuration and utilities
pip install easydict
pip install yaml
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

### 2. Training Pipeline

The training process consists of three stages:

#### Stage 1: CLIP Pre-training

Train the CLIP model to align mmWave radar signals with text descriptions:

```bash
# Train CLIP with distributed training (2 GPUs)
torchrun --nproc_per_node=2 train_clip.py \
    --model-config config/model/clip.yaml \
    --data-config config/data/humanml3d.yaml
```

**Output**: The model checkpoints will be saved to `log/humanml3d_experiments/YYYYMMDD_HHMMSS_clip/checkpoints/`

**Note**: Replace `YYYYMMDD_HHMMSS` with the actual timestamp generated during training.

#### Stage 2: Extract Pre-trained Radar Encoder

Extract the trained radar encoder weights for LLM fine-tuning:

```bash
python tools/extrack_model_weight.py \
    --checkpoint log/humanml3d_experiments/YYYYMMDD_HHMMSS_clip/checkpoints/last.ckpt
```

**Output**: The radar encoder will be saved to `feature/YYYYMMDD_HHMMSS_clip/` containing:
- `radar_encoder.pth`: Pre-trained radar encoder weights
- `radar_encoder_config.yaml`: Encoder configuration file

#### Stage 3: LLM Fine-tuning

Fine-tune the LLM (Phi-3) with the extracted radar encoder features:

```bash
deepspeed --include localhost:0 --master_port 1234 \
    train_llm.py \
    --config config/llm/phi3.yaml \
    --data_root feature/YYYYMMDD_HHMMSS_clip \
    --batch_size 12 \
    --num_workers 4 \
    --max_epochs 3 \
    --gradient_accumulation_steps 1 \
    --zero_stage 1 \
    --dtype bf16 \
    --train_split "dataset/HumanML3D/_split/train_QAs.json" \
    --test_split "dataset/HumanML3D/_split/test_QAs.json" \
    --use_random_question_for_caption true
```

**Output**: The fine-tuned model will be saved to `output/YYYYMMDD_HHMMSS_phi3/` containing:
- `adapter_model.safetensors`: LoRA adapter weights
- `adapter_config.json`: LoRA configuration
- `non_lora_trainables.safetensors`: Non-LoRA trainable weights
- `training_config.json`: Training configuration

### 3. Evaluation

Evaluate the fine-tuned LLM model:

```bash
deepspeed --include localhost:0,1 --master_port 1234 \
    evaluate_llm.py \
    --model_checkpoint output/YYYYMMDD_HHMMSS_phi3 \
    --config config/llm/phi3.yaml \
    --data_root feature/YYYYMMDD_HHMMSS_clip \
    --batch_size 12 \
    --test_split "dataset/HumanML3D/_split/test_QAs.json" \
    --use_random_question_for_caption true
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
# Evaluate fine-tuned LLM model using DeepSpeed
deepspeed --include localhost:0,1 --master_port 1234 \
    evaluate_llm.py \
    --model_checkpoint output/YYYYMMDD_HHMMSS_phi3 \
    --config config/llm/phi3.yaml \
    --data_root feature/YYYYMMDD_HHMMSS_clip \
    --batch_size 12 \
    --test_split "dataset/HumanML3D/_split/test_QAs.json" \
    --use_random_question_for_caption true

# Run comprehensive language evaluation (if available)
python tmp/evaluation/LanguageEvaluator.py --model_path path/to/model --data_path path/to/test_data

# GPT-based evaluation (for qualitative assessment)
python tmp/evaluation/GPTEvaluator.py --model_path path/to/model --data_path path/to/test_data
```

## 🛠️ Advanced Usage

### Custom Training

```python
# Custom dataset integration
from src.llm.dataset import WaveLLMDataset

# Create custom dataset
dataset = WaveLLMDataset(
    data_root="path/to/your/data",
    split="train",
    tokenizer=tokenizer
)

# Custom training using train_llm.py with command-line arguments
# All training parameters can be overridden via command line
```

### Model Customization

```python
# Custom radar encoder
from src.clip.encoders.radar_encoder_vit import RadarEncoderViT

custom_encoder = RadarEncoderViT(
    vit_model='vit_small_patch16_224.augreg_in21k',
    embed_dim=768,
    patch_size_range=[32, 16],
    patch_size_other=[16, 16],
    max_sequence_length=248
)

# Custom text encoder
from src.clip.encoders.text_encoder import TextEncoder

custom_text_encoder = TextEncoder(
    model_name='sentence-transformers/all-mpnet-base-v2',
    embed_dim=768,
    max_length=77,
    freeze_backbone=False,
    unfreeze_last_layers=1
)
```

## 🛠️ Development Tools

### Data Visualization
```bash
# Visualize radar data with heatmaps and motion plots
python tools/view_data.py --config config/data/humanml3d.yaml --num_samples 5
```

### Model Inspection
```bash
# Inspect text encoder architecture and freezing strategies
python tools/inspect_text_encoder.py --config config/model/clip.yaml

# Preview freezing strategies before training
python tools/preview_freezing_strategies.py --experiment-dir config/model/experiments-freeze-layers/
```

### Configuration Validation
```bash
# Test configuration files for validity (dry run mode)
python train_clip.py \
    --model-config config/model/clip.yaml \
    --data-config config/data/humanml3d.yaml \
    --dry-run
```

## 📁 Project Structure

```
mmExpert/
├── README.md                   # Project documentation
├── train_clip.py              # CLIP pre-training main entry point
├── train_llm.py              # LLM fine-tuning main entry point
├── evaluate_llm.py           # LLM evaluation script
├── evaluate_clip.py          # CLIP evaluation script
├── src/                       # Source code directory
│   ├── clip/                  # CLIP component
│   │   ├── clip/             # CLIP model implementations
│   │   │   ├── clip_model.py # Main CLIP model (LightningModule)
│   │   │   ├── clip_loss.py  # CLIP/SigLIP loss functions
│   │   │   ├── clip_transformer.py # Transformer components
│   │   │   └── sequence_similarity.py # Sequence similarity computation
│   │   ├── core/             # Core abstractions and base classes
│   │   │   ├── base.py       # Base classes for encoders, models, data
│   │   │   ├── registry.py   # Component registry
│   │   │   ├── factory.py   # Factory for instantiation
│   │   │   ├── injection.py # Dependency injection
│   │   │   └── pipeline.py  # Pipeline implementation
│   │   ├── encoders/         # Encoders for different modalities
│   │   │   ├── radar_encoder_vit.py # ViT-based radar encoder
│   │   │   ├── radar_encoder_temporal.py # Temporal radar encoder
│   │   │   └── text_encoder.py # Text encoder
│   │   ├── datamodule.py     # PyTorch Lightning data module
│   │   ├── dataset.py        # Dataset implementations
│   │   └── utils/            # Utility functions
│   │       ├── io.py         # I/O utilities
│   │       ├── tools.py      # General utilities
│   │       └── config_printer.py # Configuration printing
│   ├── llm/                  # LLM component
│   │   ├── llm/             # LLM model implementations
│   │   │   ├── modeling_causal.py # Main causal LLM model
│   │   │   ├── model_factory.py # Model factory
│   │   │   └── utils.py      # LLM utilities
│   │   ├── datamodule.py    # PyTorch Lightning data module
│   │   ├── dataset.py       # Dataset implementations
│   │   ├── trainer.py       # PyTorch Lightning trainer
│   │   └── utils/           # Utility functions
│   │       ├── config_loader.py # Configuration loader
│   │       ├── common_utils.py # Common utilities
│   │       ├── deepspeed_utils.py # DeepSpeed utilities
│   │       ├── conversation.py # Conversation templates
│   │       ├── trainer_lora_utils.py # LoRA utilities
│   │       └── trainer_debug_utils.py # Debug utilities
│   └── logger.py            # Logging utilities
├── config/                    # Configuration files
│   ├── data/                 # Data configurations
│   │   └── humanml3d.yaml    # HumanML3D dataset configuration
│   ├── model/                # Model configurations
│   │   ├── clip.yaml         # CLIP model configuration
│   │   ├── siglip.yaml       # SigLIP model configuration
│   │   └── experiments-clip.yaml # CLIP experiment configurations
│   └── llm/                  # LLM configurations
│       ├── phi3.yaml         # Phi-3 model configuration
│       ├── phi4.yaml         # Phi-4 model configuration
│       └── qwen3.yaml        # Qwen3 model configuration
├── dataset/                   # Dataset storage
│   ├── HumanML3D/           # HumanML3D dataset structure
│   │   ├── _split/          # Dataset splits
│   │   │   ├── train.json   # Training split
│   │   │   ├── test.json    # Test split
│   │   │   ├── train_QAs.json # Training Q&A pairs
│   │   │   └── test_QAs.json  # Test Q&A pairs
│   │   └── mmwave.zip       # mmWave radar data
│   └── HumanML3DExt/         # Extended HumanML3D dataset
├── tools/                     # Development and analysis tools
│   ├── view_data.py         # Data visualization tool
│   └── extrack_model_weight.py # Extract radar encoder weights
├── huggingface/               # Cached HuggingFace models (offline mode)
├── log/                       # Training logs and checkpoints
│   └── humanml3d_experiments/ # CLIP training outputs
│       └── YYYYMMDD_HHMMSS_clip/ # Timestamped experiment folders
│           ├── checkpoints/  # Model checkpoints
│           └── config/      # Saved configuration files
├── output/                    # LLM training outputs
│   └── YYYYMMDD_HHMMSS_phi3/ # Timestamped model folders
│       ├── adapter_model.safetensors # LoRA adapter weights
│       ├── adapter_config.json # LoRA configuration
│       └── training_config.json # Training configuration
├── feature/                   # Feature outputs and embeddings
│   └── YYYYMMDD_HHMMSS_clip/ # Extracted radar encoder
│       ├── radar_encoder.pth # Encoder weights
│       └── radar_encoder_config.yaml # Encoder configuration
├── swanlog/                   # SwanLab experiment logs
└── doc/                       # Additional documentation
```

## 🔍 Key Features

### Experiment Management
- **Configuration Printing**: Automatic display of core parameters before training
- **Batch Evaluation**: Comprehensive evaluation with multiple metrics
- **Checkpoint Management**: Automatic saving and loading of model checkpoints
- **SwanLab Integration**: Real-time experiment tracking and visualization

### Advanced Training Features
- **Multi-GPU Training**: Distributed training with DDP strategy
- **Offline Model Support**: Large model cache for offline training
- **Adaptive Patch Sizing**: Dynamic patch sizes for different radar resolutions
- **Multi-View Processing**: Simultaneous processing of range, doppler, and azimuth views

### Development and Debugging
- **Rich Visualization Tools**: OpenCV-based data visualization with consistent colormaps
- **Configuration Validation**: Pre-training validation of configuration files
- **Model Inspection**: Detailed analysis of encoder architectures and freezing strategies
- **Performance Monitoring**: Real-time tracking of training and evaluation metrics
