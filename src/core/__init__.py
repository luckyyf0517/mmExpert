"""
Core abstractions and infrastructure for mmExpert framework.

This module provides the fundamental building blocks for the framework including:
- Abstract base classes for models, encoders, processors, datasets, and losses
- Registry system for component discovery and management
- Factory pattern for component creation
- Configuration management with validation
- Dependency injection system
- Pipeline system for data processing workflows
"""

from .base import (
    # Core abstractions
    BaseEncoder,
    BaseProcessor,
    BaseDataset,
    BaseLoss,
    BaseModel,
    BasePipeline,
    BaseConfig,
    BaseFactory,

    # Data structures
    ModalityType,
    ModalityData,
    EncodingResult,

    # Utility functions
    create_modality_data,
    validate_modalities
)

from .registry import (
    # Registry system
    ComponentRegistry,
    RegistrationInfo,
    registry,

    # Decorators
    register_component,
    register_encoder,
    register_model,
    register_processor,
    register_loss
)

from .factory import (
    # Factory system
    ComponentFactory,
    EncoderFactory,
    ModelFactory,
    ProcessorFactory,
    LossFactory,
    AutoFactory,

    # Config classes
    FactoryConfig,

    # Global factory instances
    encoder_factory,
    model_factory,
    processor_factory,
    loss_factory,
    auto_factory,

    # Convenience functions
    create_component,
    create_encoder,
    create_model,
    create_processor,
    create_loss
)

from .config import (
    # Configuration system
    BaseValidatedConfig,
    ConfigValidator,
    ValidationRule,
    ConfigFormat,
    ConfigValidationError,
    ConfigManager,

    # Configuration classes
    ModelConfig,
    EncoderConfig,
    TrainingConfig,
    DatasetConfig,
    ExperimentConfig,

    # Global config manager
    config_manager,

    # Convenience functions
    load_config,
    save_config,
    create_config_template
)

from .injection import (
    # Dependency injection
    ServiceLifetime,
    ServiceDescriptor,
    DIContainer,
    DIScope,
    Injectable,
    InjectionError,
    CircularDependencyError,

    # Global container
    get_container,

    # Decorators
    injectable,

    # Functions
    inject,
    register_singleton,
    register_transient,
    register_scoped,
    resolve,
    create_scope
)

from .pipeline import (
    # Pipeline system
    ExecutionMode,
    PipelineContext,
    PipelineStep,
    PipelineExecutor,
    BasePipelineImpl,
    ProcessingPipeline,
    ModelPipeline,

    # Processors
    ModelEncodingProcessor,
    ConditionalProcessor,

    # Builder
    PipelineBuilder,

    # Convenience functions
    create_pipeline,
    sequential_pipeline,
    parallel_pipeline,
    conditional_pipeline
)

__all__ = [
    # Base abstractions
    'BaseEncoder',
    'BaseProcessor',
    'BaseDataset',
    'BaseLoss',
    'BaseModel',
    'BasePipeline',
    'BaseConfig',
    'BaseFactory',

    # Data structures
    'ModalityType',
    'ModalityData',
    'EncodingResult',

    # Registry
    'ComponentRegistry',
    'RegistrationInfo',
    'registry',
    'register_component',
    'register_encoder',
    'register_model',
    'register_processor',
    'register_loss',

    # Factory
    'ComponentFactory',
    'EncoderFactory',
    'ModelFactory',
    'ProcessorFactory',
    'LossFactory',
    'AutoFactory',
    'FactoryConfig',
    'encoder_factory',
    'model_factory',
    'processor_factory',
    'loss_factory',
    'auto_factory',
    'create_component',
    'create_encoder',
    'create_model',
    'create_processor',
    'create_loss',

    # Configuration
    'BaseValidatedConfig',
    'ConfigValidator',
    'ValidationRule',
    'ConfigFormat',
    'ConfigValidationError',
    'ConfigManager',
    'ModelConfig',
    'EncoderConfig',
    'TrainingConfig',
    'DatasetConfig',
    'ExperimentConfig',
    'config_manager',
    'load_config',
    'save_config',
    'create_config_template',

    # Dependency injection
    'ServiceLifetime',
    'ServiceDescriptor',
    'DIContainer',
    'DIScope',
    'Injectable',
    'InjectionError',
    'CircularDependencyError',
    'get_container',
    'injectable',
    'inject',
    'register_singleton',
    'register_transient',
    'register_scoped',
    'resolve',
    'create_scope',

    # Pipeline
    'ExecutionMode',
    'PipelineContext',
    'PipelineStep',
    'PipelineExecutor',
    'BasePipelineImpl',
    'ProcessingPipeline',
    'ModelPipeline',
    'ModelEncodingProcessor',
    'ConditionalProcessor',
    'PipelineBuilder',
    'create_pipeline',
    'sequential_pipeline',
    'parallel_pipeline',
    'conditional_pipeline',

    # Utilities
    'create_modality_data',
    'validate_modalities'
]