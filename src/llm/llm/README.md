# Base Model Wrapping Paradigm

This document describes the architectural pattern for wrapping base language models with wave (mmwave) feature support.

## Architecture Overview

The codebase follows a two-layer abstraction pattern for multimodal model integration:

### 1. Base Layer: `WaveModelBase`

**Purpose**: Provides core wave feature processing capabilities

**Inheritance**: `PreTrainedModel, ABC`

**Responsibilities**:
- Abstract base class definition for all wave-enabled models
- Core wave feature processing logic (model-specific)

#### CausalLM Models (`base_model_casual.py`)
- **`process_wave_features()`**: Replaces wave_patch_tokens in input_ids with wave features
- **Token-based integration**: Wave features are embedded into text sequence
- **Input processing**: `input_ids` contains special wave tokens that get replaced

#### Vision2Seq Models (`base_model_vision.py`)
- **Direct multimodal input**: Wave features passed as separate modality
- **No token replacement**: `input_ids` remains unchanged, wave features processed separately
- **Input processing**: `input_features` used for wave-visual fusion

### 2. Wrapper Layer: Model-Specific Base Classes

**Purpose**: Abstract wrapper for different model architectures

#### For Causal Language Models: `WaveModelForCausalBase`
- **Inheritance**: `WaveModelBase, ABC` (inherits from local WaveModelBase)
- **Use Case**: Text-only models with wave feature integration
- **Functionality**: Token-based wave feature injection
- **Loss computation**: Handles causal language modeling loss

#### For Vision2Seq Models: `WaveModelForVision2SeqBase`
- **Inheritance**: `WaveModelBase, ABC` (inherits from local WaveModelBase)
- **Use Case**: Wave-to-text models (no image input)
- **Functionality**: Direct multimodal fusion using `input_features`
- **Configuration**: `pixel_values=None` (image-free operation)

## Key Differences Between Model Types

### CausalLM Models (Text + Wave)
```python
# Wave features embedded into text sequence
outputs = model(
    input_ids=input_ids_with_wave_tokens,  # Contains wave_patch_token
    input_wave_embeds=wave_features,        # [B, T, H] already projected
    attention_mask=attention_mask
)
# Process: wave_patch_token -> wave_feature_embedding
```

### Vision2Seq Models (Wave → Text)
```python
# Wave features as separate modality
outputs = model(
    input_ids=text_only,               # Pure text tokens
    input_wave_embeds=wave_features,   # [B, N_w, H] already projected
    pixel_values=None,                 # No image input
    attention_mask=attention_mask
)
# Process: input_features + text -> multimodal fusion
```

## Wave Feature Processing

### Projection Layer Management
Both wrapper classes manage `mm_projection_layers`:

```python
def initialize_wave_projection(self, wave_feature_dim: int):
    """Initialize projection layer from wave dim to model hidden size"""
    # Creates: Linear(wave_feature_dim -> hidden_size)
    self.mm_projection_layers = nn.Linear(wave_feature_dim, hidden_size)
```

### Input Assumptions
- **CausalLM**: `input_wave_embeds` are already projected via `mm_projection_layers`
- **Vision2Seq**: `input_wave_embeds` are already projected and ready for multimodal fusion

## Key Design Principles

### 1. **Architecture-Specific Processing**
- CausalLM: Token replacement approach
- Vision2Seq: Direct multimodal input approach

### 2. **Pre-processed Features**
- Models assume `input_wave_embeds` are pre-projected
- No projection logic in forward passes
- Clear separation between preprocessing and inference

### 3. **Direct Inheritance Over Hooks**
- Avoids unstable hook mechanisms
- Uses clean inheritance and method overriding
- Predictable forward pass behavior

### 4. **Factory Pattern Integration**
- Dynamic class generation in `model_factory.py`
- Support for multiple model architectures (Phi, Qwen, Vision2Seq)

## Usage Patterns

### CausalLM Usage
```python
# 1. Create and initialize model
model = create_wave_causal_model(base_class, config)
model.initialize_wave_projection(wave_feature_dim=1024)

# 2. Prepare input with wave tokens
input_ids = insert_wave_tokens(text_tokens, num_wave_tokens)

# 3. Forward pass
wave_features = model.mm_projection_layers(raw_wave_features)
outputs = model(
    input_ids=input_ids,
    input_wave_embeds=wave_features,
    attention_mask=attention_mask,
    labels=labels
)
```

### Vision2Seq Usage
```python
# 1. Create and initialize model
model = create_wave_vision_model(base_class, config)
model.initialize_wave_projection(wave_feature_dim=1024)

# 2. Pre-project wave features
wave_features = model.mm_projection_layers(raw_wave_features)

# 3. Forward pass (wave-only, no images)
outputs = model(
    input_ids=text_tokens,           # Pure text input
    input_wave_embeds=wave_features, # [B, N_w, H]
    pixel_values=None,               # No image input
    attention_mask=attention_mask,
    labels=labels
)
```

## File Structure

```
base_model_casual.py     # CausalLM models with wave token integration
├── WaveModelBase       # Token replacement logic
└── WaveModelForCausalBase  # CausalLM wrapper + loss computation

base_model_vision.py     # Vision2Seq models adapted for wave-only input
├── WaveModelBase       # Direct multimodal input logic
└── WaveModelForVision2SeqBase  # Wave-to-text wrapper

model_factory.py        # Dynamic class generation
```

This paradigm ensures maintainable, extensible, and consistent multimodal model integration while respecting the architectural differences between text-wave and wave-text processing approaches.