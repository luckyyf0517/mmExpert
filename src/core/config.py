"""
Configuration management system for mmExpert framework.

This module provides a comprehensive configuration system with validation,
type safety, and environment-specific management.
"""

import os
import yaml
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import logging

from .base import BaseConfig


T = TypeVar('T')


class ConfigFormat(Enum):
    """Supported configuration formats."""
    YAML = "yaml"
    JSON = "json"
    DICT = "dict"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class ValidationRule:
    """Rule for validating configuration fields."""
    field_name: str
    required: bool = True
    field_type: Optional[Type] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[callable] = None
    description: str = ""


class ConfigValidator:
    """Validates configuration against defined rules."""

    def __init__(self, rules: List[ValidationRule]):
        self.rules = {rule.field_name: rule for rule in rules}
        self.logger = logging.getLogger(__name__)

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against rules.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validated configuration (possibly with default values)

        Raises:
            ConfigValidationError: If validation fails
        """
        validated_config = {}
        errors = []

        # Check all rules
        for field_name, rule in self.rules.items():
            try:
                value = self._validate_field(field_name, config.get(field_name), rule)
                validated_config[field_name] = value
            except ConfigValidationError as e:
                errors.append(str(e))

        # Check for unknown fields
        known_fields = set(self.rules.keys())
        unknown_fields = set(config.keys()) - known_fields
        if unknown_fields:
            self.logger.warning(f"Unknown configuration fields: {unknown_fields}")
            # Still include unknown fields in validated config
            for field in unknown_fields:
                validated_config[field] = config[field]

        if errors:
            raise ConfigValidationError(f"Configuration validation failed:\n" + "\n".join(errors))

        return validated_config

    def _validate_field(self, field_name: str, value: Any, rule: ValidationRule) -> Any:
        """Validate a single field against its rule."""
        # Check if required
        if rule.required and value is None:
            raise ConfigValidationError(f"Field '{field_name}' is required")

        # Skip validation if value is None and not required
        if value is None and not rule.required:
            return None

        # Type validation
        if rule.field_type and not isinstance(value, rule.field_type):
            try:
                # Try to convert type
                value = rule.field_type(value)
            except (ValueError, TypeError):
                raise ConfigValidationError(
                    f"Field '{field_name}' must be of type {rule.field_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        # Value range validation
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                raise ConfigValidationError(
                    f"Field '{field_name}' must be >= {rule.min_value}, got {value}"
                )
            if rule.max_value is not None and value > rule.max_value:
                raise ConfigValidationError(
                    f"Field '{field_name}' must be <= {rule.max_value}, got {value}"
                )

        # Allowed values validation
        if rule.allowed_values and value not in rule.allowed_values:
            raise ConfigValidationError(
                f"Field '{field_name}' must be one of {rule.allowed_values}, got {value}"
            )

        # Custom validation
        if rule.custom_validator:
            try:
                result = rule.custom_validator(value)
                if result is not None:
                    value = result
            except Exception as e:
                raise ConfigValidationError(
                    f"Custom validation failed for field '{field_name}': {e}"
                )

        return value


class BaseValidatedConfig(BaseConfig):
    """
    Base class for validated configurations.

    Provides automatic validation based on class annotations and rules.
    """

    # Subclasses should override this with their validation rules
    validation_rules: List[ValidationRule] = []

    def __init__(self, **kwargs):
        # Convert to dict for validation
        config_dict = self._prepare_config_dict(kwargs)

        # Validate configuration
        if self.validation_rules:
            validator = ConfigValidator(self.validation_rules)
            validated_dict = validator.validate(config_dict)
            self._validated_config = validated_dict
        else:
            self._validated_config = config_dict

        super().__init__(**self._validated_config)

    def _prepare_config_dict(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration dictionary for validation."""
        # Get default values from dataclass fields
        config_dict = {}
        if hasattr(self, '__dataclass_fields__'):
            for field_name, field_info in self.__dataclass_fields__.items():
                if field_info.default is not field_info.default_factory:
                    config_dict[field_name] = field_info.default

        # Update with provided kwargs
        config_dict.update(kwargs)
        return config_dict

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Base validation - overridden by validation rules."""
        # This is handled by the validation system above
        pass


@dataclass
class ModelConfig(BaseValidatedConfig):
    """Configuration for models."""
    name: str
    embed_dim: int = 512
    modality_types: List[str] = field(default_factory=lambda: ["radar", "text"])
    learning_rate: float = 1e-4
    max_epochs: int = 50
    temperature: float = 0.07
    use_siglip: bool = False

    validation_rules = [
        ValidationRule("name", required=True, field_type=str),
        ValidationRule("embed_dim", required=True, field_type=int, min_value=1),
        ValidationRule("learning_rate", required=False, field_type=float, min_value=0, max_value=1),
        ValidationRule("max_epochs", required=False, field_type=int, min_value=1),
        ValidationRule("temperature", required=False, field_type=float, min_value=0),
        ValidationRule("use_siglip", required=False, field_type=bool),
    ]


@dataclass
class EncoderConfig(BaseValidatedConfig):
    """Configuration for encoders."""
    name: str
    embed_dim: int = 512
    input_dim: Optional[int] = None
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    validation_rules = [
        ValidationRule("name", required=True, field_type=str),
        ValidationRule("embed_dim", required=True, field_type=int, min_value=1),
        ValidationRule("input_dim", required=False, field_type=int, min_value=1),
        ValidationRule("num_layers", required=False, field_type=int, min_value=1),
        ValidationRule("num_heads", required=False, field_type=int, min_value=1),
        ValidationRule("dropout", required=False, field_type=float, min_value=0, max_value=1),
    ]


@dataclass
class TrainingConfig(BaseValidatedConfig):
    """Configuration for training."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 50
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    seed: int = 42

    validation_rules = [
        ValidationRule("batch_size", required=False, field_type=int, min_value=1),
        ValidationRule("learning_rate", required=False, field_type=float, min_value=0),
        ValidationRule("max_epochs", required=False, field_type=int, min_value=1),
        ValidationRule("gradient_clip_val", required=False, field_type=float, min_value=0),
        ValidationRule("accumulate_grad_batches", required=False, field_type=int, min_value=1),
        ValidationRule("num_workers", required=False, field_type=int, min_value=0),
        ValidationRule("seed", required=False, field_type=int),
    ]


@dataclass
class DatasetConfig(BaseValidatedConfig):
    """Configuration for datasets."""
    name: str
    data_path: str
    split_file: Optional[str] = None
    max_motion_length: int = 496
    min_motion_len: int = 96
    unit_length: int = 16
    normalize: str = "per_frame"
    radar_views: str = "all"

    validation_rules = [
        ValidationRule("name", required=True, field_type=str),
        ValidationRule("data_path", required=True, field_type=str),
        ValidationRule("split_file", required=False, field_type=str),
        ValidationRule("max_motion_length", required=False, field_type=int, min_value=1),
        ValidationRule("min_motion_len", required=False, field_type=int, min_value=1),
        ValidationRule("unit_length", required=False, field_type=int, min_value=1),
        ValidationRule("normalize", required=False, field_type=str,
                      allowed_values=["none", "per_frame", "global", "log"]),
        ValidationRule("radar_views", required=False, field_type=str,
                      allowed_values=["all", "doppler_only", "range_only", "azimuth_only"]),
    ]


@dataclass
class ExperimentConfig(BaseValidatedConfig):
    """Complete experiment configuration."""
    name: str
    description: str = ""
    model: ModelConfig = None
    dataset: DatasetConfig = None
    training: TrainingConfig = None
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500

    def __post_init__(self):
        # Create default configs if not provided
        if self.model is None:
            self.model = ModelConfig(name="default")
        if self.dataset is None:
            self.dataset = DatasetConfig(name="default", data_path="")
        if self.training is None:
            self.training = TrainingConfig()

        # Validate after creating defaults
        if not hasattr(self, '_validated'):
            super().__init__(**asdict(self))
            self._validated = True


class ConfigManager:
    """
    Manages configuration loading, saving, and environment handling.

    Provides:
    - Multi-format loading (YAML, JSON, dict)
    - Environment-specific configurations
    - Configuration merging and inheritance
    - Template generation
    """

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.logger = logging.getLogger(__name__)

    def load_config(self,
                   config_path: Union[str, Path],
                   format: Optional[ConfigFormat] = None,
                   environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
            format: Configuration format (auto-detected if not provided)
            environment: Environment-specific override (dev, test, prod)

        Returns:
            Loaded configuration dictionary
        """
        config_path = Path(config_path)

        # Auto-detect format if not provided
        if format is None:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                format = ConfigFormat.YAML
            elif config_path.suffix.lower() == '.json':
                format = ConfigFormat.JSON
            else:
                raise ValueError(f"Cannot determine format for {config_path}")

        # Load base configuration
        config_dict = self._load_file(config_path, format)

        # Apply environment-specific overrides
        if environment:
            env_config = self._load_environment_config(config_path, environment)
            config_dict = self._merge_configs(config_dict, env_config)

        return config_dict

    def save_config(self,
                   config: Dict[str, Any],
                   config_path: Union[str, Path],
                   format: ConfigFormat = ConfigFormat.YAML) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary to save
            config_path: Output path
            format: Output format
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ConfigFormat.YAML:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif format == ConfigFormat.JSON:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format for saving: {format}")

        self.logger.info(f"Configuration saved to {config_path}")

    def create_template(self,
                       template_type: str,
                       output_path: Union[str, Path],
                       format: ConfigFormat = ConfigFormat.YAML) -> None:
        """
        Create a configuration template.

        Args:
            template_type: Type of template (model, dataset, experiment)
            output_path: Output path for template
            format: Output format
        """
        templates = {
            "model": asdict(ModelConfig(name="template")),
            "dataset": asdict(DatasetConfig(name="template", data_path="/path/to/data")),
            "training": asdict(TrainingConfig()),
            "experiment": asdict(ExperimentConfig(
                name="template_experiment",
                description="Template experiment configuration"
            ))
        }

        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")

        self.save_config(templates[template_type], output_path, format)

    def _load_file(self, file_path: Path, format: ConfigFormat) -> Dict[str, Any]:
        """Load configuration file in specified format."""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            if format == ConfigFormat.YAML:
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            elif format == ConfigFormat.JSON:
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading {file_path}: {e}")

    def _load_environment_config(self, base_path: Path, environment: str) -> Dict[str, Any]:
        """Load environment-specific configuration override."""
        env_path = base_path.parent / f"{base_path.stem}_{environment}{base_path.suffix}"

        if env_path.exists():
            return self._load_file(env_path, ConfigFormat.YAML if env_path.suffix in ['.yaml', '.yml'] else ConfigFormat.JSON)
        else:
            self.logger.info(f"Environment config not found: {env_path}")
            return {}

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result


# Global config manager instance
config_manager = ConfigManager()


# Convenience functions
def load_config(config_path: Union[str, Path],
               environment: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    config_dict = config_manager.load_config(config_path, environment=environment)

    # Resolve data config
    if isinstance(config_dict.get('data_cfg'), str):
        data_path = config_dict['data_cfg']
        if not os.path.isabs(data_path):
            data_path = os.path.join(os.path.dirname(str(config_path)), data_path)
        config_dict['data_cfg'] = config_manager.load_config(data_path)

    # Resolve model config
    if isinstance(config_dict.get('model_cfg'), str):
        model_path = config_dict['model_cfg']
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(str(config_path)), model_path)
        config_dict['model_cfg'] = config_manager.load_config(model_path)

    return config_dict


def save_config(config: Dict[str, Any],
               config_path: Union[str, Path],
               format: ConfigFormat = ConfigFormat.YAML) -> None:
    """Save configuration to file."""
    config_manager.save_config(config, config_path, format)


def create_config_template(template_type: str,
                          output_path: Union[str, Path],
                          format: ConfigFormat = ConfigFormat.YAML) -> None:
    """Create configuration template."""
    config_manager.create_template(template_type, output_path, format)