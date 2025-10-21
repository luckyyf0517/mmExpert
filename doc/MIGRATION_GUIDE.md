# Migration Guide: From Original to Refactored Architecture

This guide helps you migrate from the original mmExpert code to the new abstraction layer.

## Overview of Changes

The refactored architecture introduces:
- **Clean abstractions**: Base classes for all major components
- **Registry system**: Plugin-style component discovery
- **Factory pattern**: Type-safe component creation
- **Configuration management**: Validated configuration with templates
- **Dependency injection**: Flexible service management
- **Pipeline system**: Modular data processing workflows

## Migration Steps

### 1. Import Changes

**Before:**
```python
from src.model.clip import CLIP
from src.model.clip_radar_encoder import RadarEncoder
from src.misc.tools import instantiate_from_config
```

**After:**
```python
from src.core import (
    create_model, create_encoder,
    ModelConfig, EncoderConfig,
    registry, auto_factory
)
from src.model.clip_model import CLIPModel
from src.encoders.radar_encoder import RadarEncoder
```

### 2. Model Creation

**Before:**
```python
# Old way using instantiate_from_config
model_cfg = {
    'target': 'src.model.clip.CLIP',
    'params': {
        'encoder_cfg': {...},
        'text_cfg': {...},
        'context_length': 77,
        'transformer_width': 512,
        'temperature': 0.07
    }
}
model = instantiate_from_config(model_cfg)
```

**After:**
```python
# New way using factories and validated configs
model_config = ModelConfig(
    name="clip_model",
    embed_dim=512,
    temperature=0.07,
    use_siglip=False
)

# Method 1: Using convenience function
model = create_model("clip_model", model_config.to_dict())

# Method 2: Using auto factory
model = auto_factory.create_model("clip_model", model_config.to_dict())

# Method 3: Direct instantiation
model = CLIPModel(**model_config.to_dict())
```

### 3. Encoder Usage

**Before:**
```python
# Direct instantiation with hardcoded parameters
radar_encoder = RadarEncoder(
    input_dim=256,
    embed_dim=512,
    num_layers=4
)
```

**After:**
```python
# Using factory with configuration
encoder_config = EncoderConfig(
    name="radar_encoder",
    embed_dim=512,
    num_layers=4,
    num_heads=8
)

radar_encoder = create_encoder("radar_encoder_v2", encoder_config.to_dict())
```

### 4. Configuration Management

**Before:**
```python
# Raw YAML/Dict configuration with no validation
config = load_yaml("config.yaml")
model = instantiate_from_config(config["model_cfg"])
```

**After:**
```python
# Validated configuration with type safety
from src.core import load_config, ExperimentConfig

# Load and validate configuration
config_dict = load_config("experiment_config.yaml")
experiment_config = ExperimentConfig(**config_dict)

# Create components from validated config
model = create_model(
    experiment_config.model.name,
    experiment_config.model.to_dict()
)
```

### 5. Data Processing

**Before:**
```python
# Manual data processing
def process_radar_data(data):
    # Hardcoded processing logic
    processed = preprocess(data)
    return processed

# Usage in training loop
for batch in dataloader:
    radar_data = process_radar_data(batch['radar'])
    text_data = batch['text']
    features = model(radar_data, text_data)
```

**After:**
```python
# Pipeline-based processing
from src.core import ProcessingPipeline

# Create processors
class RadarProcessor(BaseProcessor):
    def process(self, data, **kwargs):
        return preprocess(data)

# Build pipeline
pipeline = ProcessingPipeline("training_pipeline")
pipeline.preprocess(RadarProcessor(), name="radar_prep")
pipeline.add_step("encoding", model)

# Usage
for batch in dataloader:
    results = pipeline.process({
        "radar_data": batch['radar'],
        "text": batch['text']
    })
```

### 6. Component Registration

**Before:**
```python
# No registration system - components were hard-coded
# To add a new component, you had to modify existing code
```

**After:**
```python
# Register new components easily
@register_encoder(
    name="my_custom_encoder",
    description="My custom radar encoder",
    tags=["radar", "custom"]
)
class MyCustomEncoder(BaseEncoder):
    def __init__(self, embed_dim=512, **kwargs):
        super().__init__(embed_dim=embed_dim, modality=ModalityType.RADAR)
        # Your implementation

    def encode(self, data, return_sequence=False, **kwargs):
        # Your encoding logic
        pass

# Component is now discoverable and creatable:
encoder = create_encoder("my_custom_encoder", {"embed_dim": 256})
```

### 7. Training Loop Migration

**Before:**
```python
# Manual training loop
def train_model(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # Manual data preparation
            radar_data = batch['radar']
            text = batch['caption']

            # Forward pass
            radar_features, text_features = model(radar_data, text)
            loss = compute_loss(radar_features, text_features)

            # Backward pass
            loss.backward()
            optimizer.step()
```

**After:**
```python
# Pipeline-based training with dependency injection
@injectable(ServiceLifetime.SINGLETON)
class TrainingService:
    def __init__(self,
                 model: BaseModel,
                 loss_fn: BaseLoss,
                 optimizer: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Use pipeline for data processing
            results = self.training_pipeline.process(batch)

            # Compute loss
            loss_dict = self.loss_fn(
                predictions=results,
                targets=batch
            )

            # Backward pass
            loss = loss_dict["total_loss"]
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

# Training service is automatically injected with dependencies
training_service = resolve(TrainingService)
```

## Configuration Migration

### Old Configuration Format
```yaml
model_cfg:
  target: "src.model.clip.CLIP"
  params:
    encoder_cfg:
      target: "src.model.clip_radar_encoder.RadarEncoder"
      params:
        embed_dim: 512
    text_cfg:
      target: "src.model.clip_text_encoder.TextEncoder"
      params:
        embed_dim: 512
    temperature: 0.07
```

### New Configuration Format
```yaml
name: "clip_experiment"
description: "CLIP training with radar and text"
model:
  name: "clip_model"
  embed_dim: 512
  temperature: 0.07
  use_siglip: false
  encoder_configs:
    radar:
      embed_dim: 512
      num_layers: 4
      num_heads: 8
      dropout: 0.1
    text:
      embed_dim: 512
      model_name: "bert-base-uncased"
      max_length: 77
training:
  batch_size: 32
  learning_rate: 0.0001
  max_epochs: 50
  gradient_clip_val: 1.0
```

## Benefits of Migration

1. **Type Safety**: Configuration validation prevents runtime errors
2. **Modularity**: Components can be easily swapped and extended
3. **Testability**: Dependency injection enables better unit testing
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Plugin system for easy feature additions
6. **Reusability**: Components can be reused across projects

## Backward Compatibility

The refactored system maintains some backward compatibility:

```python
# You can still use old-style instantiation for gradual migration
from src.core.factory import ComponentFactory

# Bridge between old and new
old_config = {
    'target': 'src.model.clip.CLIP',
    'params': {...}
}

# Convert to new format
component_type = old_config['target'].split('.')[-1]
factory_config = FactoryConfig(
    component_type=component_type.lower() + "_v2",
    override_params=old_config['params']
)

new_component = ComponentFactory.create(factory_config)
```

## Testing Migration

When migrating, ensure you:

1. **Test component creation**: Verify all components can be instantiated
2. **Test configuration loading**: Ensure configs validate correctly
3. **Test pipeline processing**: Verify data flows through pipelines
4. **Test model outputs**: Ensure model produces expected results
5. **Test training**: Verify training loop works correctly

## Getting Help

- Check the examples in `examples/refactored_training_example.py`
- Look at the component implementations in `src/models/`, `src/encoders/`
- Review the core abstractions in `src/core/`
- Use the registry to discover available components:
  ```python
  from src.core import registry
  print(registry.list_components())
  print(registry.search(query="encoder"))
  ```