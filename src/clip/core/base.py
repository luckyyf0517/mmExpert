"""
Core abstractions for mmExpert framework.

This module defines the fundamental interfaces and abstractions that provide
a clean, extensible architecture for multimodal deep learning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from dataclasses import dataclass
from enum import Enum


class ModalityType(Enum):
    """Supported data modalities."""
    RADAR = "radar"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"


@dataclass
class ModalityData:
    """Container for multimodal data."""
    data: torch.Tensor
    modality: ModalityType
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if isinstance(self.modality, str):
            self.modality = ModalityType(self.modality)


@dataclass
class EncodingResult:
    """Result of encoding operation."""
    features: torch.Tensor
    sequence_features: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all encoders.

    This class defines the interface that all encoders must implement,
    providing a consistent way to encode different modalities.
    """

    def __init__(self,
                 embed_dim: int,
                 modality: ModalityType,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality = modality
        self._supports_sequence = False

    @abstractmethod
    def encode(self,
               data: ModalityData,
               return_sequence: bool = False,
               **kwargs) -> EncodingResult:
        """
        Encode input data into feature representations.

        Args:
            data: Input modality data
            return_sequence: Whether to return sequence-level features
            **kwargs: Additional encoder-specific arguments

        Returns:
            EncodingResult containing encoded features
        """
        pass

    @property
    def supports_sequence(self) -> bool:
        """Whether this encoder supports sequence-level encoding."""
        return self._supports_sequence

    @property
    def device(self) -> torch.device:
        """Get the device of this encoder."""
        return next(self.parameters()).device


class BaseProcessor(ABC):
    """
    Abstract base class for data processors.

    Processors handle data preprocessing, augmentation, and transformation
    in a pipeline-friendly manner.
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process input data.

        Args:
            data: Input data to process
            **kwargs: Additional processing arguments

        Returns:
            Processed data
        """
        pass

    def __call__(self, data: Any, **kwargs) -> Any:
        """Make processor callable."""
        return self.process(data, **kwargs)


class BaseDataset(ABC):
    """
    Abstract base class for datasets.

    Provides a consistent interface for different types of datasets
    while maintaining flexibility for specific implementations.
    """

    def __init__(self,
                 name: str,
                 modality_types: List[ModalityType],
                 **kwargs):
        self.name = name
        self.modality_types = modality_types
        self.config = kwargs

    @abstractmethod
    def get_item(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Dictionary containing the item data
        """
        pass

    @abstractmethod
    def get_length(self) -> int:
        """Get the length of the dataset."""
        pass

    def __len__(self) -> int:
        return self.get_length()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get_item(idx)


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for loss functions.

    Provides a consistent interface for different loss computations
    including multimodal and sequence-based losses.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def compute_loss(self,
                     predictions: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor],
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss values.

        Args:
            predictions: Model predictions
            targets: Target values
            **kwargs: Additional loss-specific arguments

        Returns:
            Dictionary of loss values
        """
        pass

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for loss computation."""
        return self.compute_loss(predictions, targets, **kwargs)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for models.

    This defines the core model interface that all models should implement,
    providing consistent methods for training, inference, and configuration.
    """

    def __init__(self,
                 name: str,
                 modality_types: List[ModalityType],
                 **kwargs):
        super().__init__()
        self.name = name
        self.modality_types = modality_types
        self.config = kwargs
        self._encoders = nn.ModuleDict()
        self._processors = {}

    @abstractmethod
    def encode(self,
               data: Dict[ModalityType, ModalityData],
               return_sequences: bool = False,
               **kwargs) -> Dict[ModalityType, EncodingResult]:
        """
        Encode multimodal data.

        Args:
            data: Dictionary mapping modalities to data
            return_sequences: Whether to return sequence-level features
            **kwargs: Additional encoding arguments

        Returns:
            Dictionary of encoding results for each modality
        """
        pass

    @abstractmethod
    def compute_similarity(self,
                          features_1: Dict[ModalityType, torch.Tensor],
                          features_2: Dict[ModalityType, torch.Tensor],
                          **kwargs) -> torch.Tensor:
        """
        Compute similarity between feature sets.

        Args:
            features_1: First set of features
            features_2: Second set of features
            **kwargs: Additional similarity arguments

        Returns:
            Similarity tensor
        """
        pass

    def add_encoder(self, modality: ModalityType, encoder: BaseEncoder):
        """Add an encoder for a specific modality."""
        self._encoders[modality.value] = encoder

    def get_encoder(self, modality: ModalityType) -> Optional[BaseEncoder]:
        """Get encoder for a specific modality."""
        return self._encoders.get(modality.value)

    def add_processor(self, name: str, processor: BaseProcessor):
        """Add a data processor."""
        self._processors[name] = processor

    def get_processor(self, name: str) -> Optional[BaseProcessor]:
        """Get a data processor by name."""
        return self._processors.get(name)

    @property
    def device(self) -> torch.device:
        """Get the device of this model."""
        return next(self.parameters()).device


class BasePipeline(ABC):
    """
    Abstract base class for processing pipelines.

    Pipelines orchestrate the end-to-end processing flow,
    from data loading to model inference.
    """

    def __init__(self,
                 name: str,
                 model: BaseModel,
                 **kwargs):
        self.name = name
        self.model = model
        self.config = kwargs
        self._processors = []

    def add_processor(self, processor: BaseProcessor):
        """Add a processor to the pipeline."""
        self._processors.append(processor)

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process data through the pipeline.

        Args:
            data: Input data
            **kwargs: Additional processing arguments

        Returns:
            Processed results
        """
        pass

    def __call__(self, data: Any, **kwargs) -> Any:
        """Make pipeline callable."""
        return self.process(data, **kwargs)


class BaseConfig(ABC):
    """
    Abstract base class for configurations.

    Provides type-safe configuration management with validation.
    """

    def __init__(self, **kwargs):
        self._validate_config(kwargs)
        self._config = kwargs

    @abstractmethod
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()


class BaseFactory(ABC):
    """
    Abstract base class for factories.

    Factories provide a consistent way to create components
    from configurations while maintaining type safety.
    """

    @classmethod
    @abstractmethod
    def create(cls, config: BaseConfig, **kwargs) -> Any:
        """
        Create an instance from configuration.

        Args:
            config: Configuration object
            **kwargs: Additional creation arguments

        Returns:
            Created instance
        """
        pass

    @classmethod
    @abstractmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported component types."""
        pass


# Utility functions for working with abstractions
def create_modality_data(data: torch.Tensor,
                        modality: Union[str, ModalityType],
                        metadata: Optional[Dict[str, Any]] = None) -> ModalityData:
    """Create a ModalityData instance."""
    return ModalityData(
        data=data,
        modality=modality if isinstance(modality, ModalityType) else ModalityType(modality),
        metadata=metadata
    )


def validate_modalities(data: Dict[ModalityType, Any],
                       expected_modalities: List[ModalityType]) -> None:
    """Validate that required modalities are present."""
    for modality in expected_modalities:
        if modality not in data:
            raise ValueError(f"Required modality {modality} not found in data")