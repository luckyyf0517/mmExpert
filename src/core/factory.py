"""
Factory system for mmExpert framework.

This module provides factory implementations for creating components
from configurations with type safety and validation.
"""

import inspect
from typing import Any, Dict, List, Type, TypeVar, Union, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .base import BaseFactory, BaseConfig, BaseModel, BaseEncoder, BaseProcessor, BaseLoss
from .registry import registry


T = TypeVar('T')


@dataclass
class FactoryConfig(BaseConfig):
    """Base configuration for factory operations."""
    component_type: str
    version: str = "latest"
    override_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after dataclass init."""
        # Initialize BaseConfig's _config attribute
        super().__init__(
            component_type=self.component_type,
            version=self.version,
            override_params=self.override_params
        )

    def _validate_config(self, config: Dict[str, Any]) -> None:
        if "component_type" not in config:
            raise ValueError("component_type is required")


class ComponentFactory(BaseFactory):
    """
    Generic factory for creating components from configurations.

    This factory provides:
    - Type-safe component creation
    - Configuration validation
    - Parameter override support
    - Registry integration
    """

    @classmethod
    def create(cls, config: FactoryConfig, **kwargs) -> Any:
        """
        Create a component from configuration.

        Args:
            config: Factory configuration
            **kwargs: Additional creation arguments

        Returns:
            Created component instance
        """
        # Get component registration info
        info = registry.get(config.component_type)
        if info is None:
            raise ValueError(f"Component '{config.component_type}' not registered")

        # Prepare creation parameters
        create_params = {}

        # Extract config parameters (excluding factory-specific fields)
        config_dict = config.to_dict()
        factory_keys = {"component_type", "version", "override_params"}
        for key, value in config_dict.items():
            if key not in factory_keys:
                create_params[key] = value

        # Apply override parameters
        create_params.update(config.override_params)
        create_params.update(kwargs)

        # Use component's factory if available
        if info.factory_class:
            return info.factory_class.create(info.config_class(**create_params) if info.config_class else create_params)

        # Use component's config class for validation if available
        if info.config_class:
            validated_config = info.config_class(**create_params)
            create_params = validated_config.to_dict()

        # Validate constructor parameters
        cls._validate_constructor_params(info.component_class, create_params)

        # Create instance
        return info.component_class(**create_params)

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported component types."""
        return registry.list_components()

    @staticmethod
    def _validate_constructor_params(component_class: Type, params: Dict[str, Any]) -> None:
        """Validate that parameters match component constructor."""
        try:
            sig = inspect.signature(component_class.__init__)
        except (ValueError, TypeError):
            # Can't inspect signature, skip validation
            return

        # Check for unknown parameters
        param_names = set(sig.parameters.keys()) - {'self'}
        unknown_params = set(params.keys()) - param_names

        if unknown_params:
            # Check if component accepts **kwargs
            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

            if not has_kwargs:
                raise ValueError(
                    f"Unknown parameters for {component_class.__name__}: {unknown_params}. "
                    f"Expected parameters: {param_names}"
                )


class EncoderFactory(ComponentFactory):
    """Factory specialized for creating encoders."""

    @classmethod
    def create(cls, config: Union[FactoryConfig, Dict[str, Any]], **kwargs) -> BaseEncoder:
        """Create an encoder instance."""
        if isinstance(config, dict):
            config = FactoryConfig(**config)

        encoder = super().create(config, **kwargs)

        if not isinstance(encoder, BaseEncoder):
            raise TypeError(f"Created component is not a BaseEncoder: {type(encoder)}")

        return encoder

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported encoder types."""
        return [name for name in registry.list_components()
                if "encoder" in registry.get(name).tags]


class ModelFactory(ComponentFactory):
    """Factory specialized for creating models."""

    @classmethod
    def create(cls, config: Union[FactoryConfig, Dict[str, Any]], **kwargs) -> BaseModel:
        """Create a model instance."""
        if isinstance(config, dict):
            config = FactoryConfig(**config)

        model = super().create(config, **kwargs)

        if not isinstance(model, BaseModel):
            raise TypeError(f"Created component is not a BaseModel: {type(model)}")

        return model

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported model types."""
        return [name for name in registry.list_components()
                if "model" in registry.get(name).tags]


class ProcessorFactory(ComponentFactory):
    """Factory specialized for creating processors."""

    @classmethod
    def create(cls, config: Union[FactoryConfig, Dict[str, Any]], **kwargs) -> BaseProcessor:
        """Create a processor instance."""
        if isinstance(config, dict):
            config = FactoryConfig(**config)

        processor = super().create(config, **kwargs)

        if not isinstance(processor, BaseProcessor):
            raise TypeError(f"Created component is not a BaseProcessor: {type(processor)}")

        return processor

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported processor types."""
        return [name for name in registry.list_components()
                if "processor" in registry.get(name).tags]


class LossFactory(ComponentFactory):
    """Factory specialized for creating loss functions."""

    @classmethod
    def create(cls, config: Union[FactoryConfig, Dict[str, Any]], **kwargs) -> BaseLoss:
        """Create a loss function instance."""
        if isinstance(config, dict):
            config = FactoryConfig(**config)

        loss = super().create(config, **kwargs)

        if not isinstance(loss, BaseLoss):
            raise TypeError(f"Created component is not a BaseLoss: {type(loss)}")

        return loss

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported loss types."""
        return [name for name in registry.list_components()
                if "loss" in registry.get(name).tags]


class AutoFactory:
    """
    Automatic factory that determines the appropriate factory based on context.

    This factory provides a convenient interface for creating components
    without explicitly specifying the factory type.
    """

    _factory_map = {
        "encoder": EncoderFactory,
        "model": ModelFactory,
        "processor": ProcessorFactory,
        "loss": LossFactory,
    }

    @classmethod
    def create(cls,
               name: str,
               config: Union[Dict[str, Any], BaseConfig] = None,
               factory_type: str = None,
               **kwargs) -> Any:
        """
        Create a component, automatically determining the factory type.

        Args:
            name: Component name
            config: Component configuration
            factory_type: Explicit factory type (optional)
            **kwargs: Additional creation arguments

        Returns:
            Created component instance
        """
        # Get component info
        info = registry.get(name)
        if info is None:
            raise ValueError(f"Component '{name}' not registered")

        # Determine factory type
        if factory_type is None:
            factory_type = cls._infer_factory_type(info)

        # Get appropriate factory
        factory = cls._factory_map.get(factory_type, ComponentFactory)

        # Prepare configuration
        if config is None:
            config = {"component_type": name}
        elif isinstance(config, BaseConfig):
            config_dict = config.to_dict()
            config_dict["component_type"] = name
            config = config_dict
        else:
            config = dict(config)
            config["component_type"] = name

        # Create component
        return factory.create(config, **kwargs)

    @classmethod
    def _infer_factory_type(cls, info) -> str:
        """Infer factory type from component registration info."""
        tags = info.tags

        if "encoder" in tags:
            return "encoder"
        elif "model" in tags:
            return "model"
        elif "processor" in tags:
            return "processor"
        elif "loss" in tags:
            return "loss"
        else:
            return "component"  # Generic factory

    @classmethod
    def create_encoder(cls, name: str, config: Dict[str, Any] = None, **kwargs) -> BaseEncoder:
        """Convenience method for creating encoders."""
        return cls.create(name, config, "encoder", **kwargs)

    @classmethod
    def create_model(cls, name: str, config: Dict[str, Any] = None, **kwargs) -> BaseModel:
        """Convenience method for creating models."""
        return cls.create(name, config, "model", **kwargs)

    @classmethod
    def create_processor(cls, name: str, config: Dict[str, Any] = None, **kwargs) -> BaseProcessor:
        """Convenience method for creating processors."""
        return cls.create(name, config, "processor", **kwargs)

    @classmethod
    def create_loss(cls, name: str, config: Dict[str, Any] = None, **kwargs) -> BaseLoss:
        """Convenience method for creating loss functions."""
        return cls.create(name, config, "loss", **kwargs)


# Global factory instances
encoder_factory = EncoderFactory()
model_factory = ModelFactory()
processor_factory = ProcessorFactory()
loss_factory = LossFactory()
auto_factory = AutoFactory()


# Convenience functions
def create_component(name: str, config: Dict[str, Any] = None, **kwargs) -> Any:
    """Create any component."""
    return auto_factory.create(name, config, **kwargs)


def create_encoder(name: str, config: Dict[str, Any] = None, **kwargs) -> BaseEncoder:
    """Create an encoder."""
    return auto_factory.create_encoder(name, config, **kwargs)


def create_model(name: str, config: Dict[str, Any] = None, **kwargs) -> BaseModel:
    """Create a model."""
    return auto_factory.create_model(name, config, **kwargs)


def create_processor(name: str, config: Dict[str, Any] = None, **kwargs) -> BaseProcessor:
    """Create a processor."""
    return auto_factory.create_processor(name, config, **kwargs)


def create_loss(name: str, config: Dict[str, Any] = None, **kwargs) -> BaseLoss:
    """Create a loss function."""
    return auto_factory.create_loss(name, config, **kwargs)