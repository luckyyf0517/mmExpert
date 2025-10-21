"""
Registry system for mmExpert framework.

This module provides a centralized registry system for managing components,
supporting plugin-style architecture and dynamic component discovery.
"""

import inspect
from typing import Any, Dict, List, Type, Optional, Callable, TypeVar, Union
from collections import defaultdict
import threading
from dataclasses import dataclass

from .base import BaseFactory, BaseConfig


T = TypeVar('T')


@dataclass
class RegistrationInfo:
    """Information about a registered component."""
    name: str
    component_class: Type
    factory_class: Optional[Type[BaseFactory]] = None
    config_class: Optional[Type[BaseConfig]] = None
    category: str = "general"
    description: str = ""
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class ComponentRegistry:
    """
    Central registry for managing framework components.

    This registry supports:
    - Component registration and discovery
    - Plugin-style architecture
    - Factory-based instantiation
    - Thread-safe operations
    - Hierarchical categories
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, '_initialized', False):
            with self._lock:
                if not getattr(self, '_initialized', False):
                    self._registry: Dict[str, Dict[str, RegistrationInfo]] = defaultdict(dict)
                    self._categories: Dict[str, List[str]] = defaultdict(list)
                    self._aliases: Dict[str, str] = {}
                    self._factories: Dict[str, Type[BaseFactory]] = {}
                    self._hooks: Dict[str, List[Callable]] = defaultdict(list)
                    self._initialized = True

    def register(self,
                 name: str,
                 component_class: Type,
                 factory_class: Optional[Type[BaseFactory]] = None,
                 config_class: Optional[Type[BaseConfig]] = None,
                 category: str = "general",
                 description: str = "",
                 tags: List[str] = None,
                 metadata: Dict[str, Any] = None,
                 aliases: List[str] = None,
                 replace: bool = False) -> None:
        """
        Register a component in the registry.

        Args:
            name: Unique name for the component
            component_class: The component class to register
            factory_class: Optional factory class for instantiation
            config_class: Optional config class for validation
            category: Category for organization
            description: Human-readable description
            tags: List of tags for searching
            metadata: Additional metadata
            aliases: List of alternative names
            replace: Whether to replace existing registration
        """
        with self._lock:
            if name in self._registry[category] and not replace:
                raise ValueError(f"Component '{name}' already registered in category '{category}'")

            # Validate component class
            if not inspect.isclass(component_class):
                raise TypeError(f"component_class must be a class, got {type(component_class)}")

            # Register component
            info = RegistrationInfo(
                name=name,
                component_class=component_class,
                factory_class=factory_class,
                config_class=config_class,
                category=category,
                description=description,
                tags=tags or [],
                metadata=metadata or {}
            )

            self._registry[category][name] = info
            self._categories[category].append(name)

            # Register factory if provided
            if factory_class:
                self._factories[f"{category}.{name}"] = factory_class

            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases and not replace:
                        raise ValueError(f"Alias '{alias}' already registered")
                    self._aliases[alias] = f"{category}.{name}"

            # Call registration hooks
            self._call_hooks("register", info)

    def get(self,
            name: str,
            category: str = None,
            default: Any = None) -> Optional[RegistrationInfo]:
        """
        Get a component registration info.

        Args:
            name: Component name or alias
            category: Category to search in (optional)
            default: Default value if not found

        Returns:
            Registration info or default value
        """
        with self._lock:
            # Check if it's an alias
            if name in self._aliases:
                full_name = self._aliases[name]
                category, name = full_name.split(".", 1)
                return self._registry[category].get(name, default)

            # Search in specific category
            if category:
                return self._registry[category].get(name, default)

            # Search all categories
            for cat_registry in self._registry.values():
                if name in cat_registry:
                    return cat_registry[name]

            return default

    def create(self,
               name: str,
               config: BaseConfig,
               category: str = None,
               **kwargs) -> Any:
        """
        Create an instance of a registered component.

        Args:
            name: Component name or alias
            config: Configuration object
            category: Category to search in (optional)
            **kwargs: Additional creation arguments

        Returns:
            Created component instance
        """
        info = self.get(name, category)
        if info is None:
            raise ValueError(f"Component '{name}' not found")

        # Use factory if available
        if info.factory_class:
            return info.factory_class.create(config, **kwargs)

        # Fall back to direct instantiation
        component_class = info.component_class
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else config

        # Validate against config class if available
        if info.config_class:
            validated_config = info.config_class(**config_dict)
            config_dict = validated_config.to_dict()

        return component_class(**config_dict, **kwargs)

    def list_categories(self) -> List[str]:
        """Get list of all registered categories."""
        with self._lock:
            return list(self._categories.keys())

    def list_components(self, category: str = None) -> List[str]:
        """Get list of all component names in a category (or all categories)."""
        with self._lock:
            if category:
                return list(self._registry.get(category, {}).keys())

            all_components = []
            for cat_components in self._registry.values():
                all_components.extend(cat_components.keys())
            return list(set(all_components))

    def find_by_tag(self, tag: str) -> List[RegistrationInfo]:
        """Find all components with a specific tag."""
        with self._lock:
            results = []
            for cat_registry in self._registry.values():
                for info in cat_registry.values():
                    if tag in info.tags:
                        results.append(info)
            return results

    def search(self,
               query: str = None,
               category: str = None,
               tags: List[str] = None) -> List[RegistrationInfo]:
        """
        Search for components.

        Args:
            query: Search query for names/descriptions
            category: Filter by category
            tags: Filter by tags

        Returns:
            List of matching component info
        """
        with self._lock:
            results = []

            for cat_name, cat_registry in self._registry.items():
                if category and cat_name != category:
                    continue

                for info in cat_registry.values():
                    # Check query match
                    if query:
                        query_lower = query.lower()
                        name_match = query_lower in info.name.lower()
                        desc_match = query_lower in info.description.lower()
                        if not (name_match or desc_match):
                            continue

                    # Check tags
                    if tags:
                        if not any(tag in info.tags for tag in tags):
                            continue

                    results.append(info)

            return results

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a hook for registry events."""
        with self._lock:
            self._hooks[event].append(callback)

    def remove_hook(self, event: str, callback: Callable) -> None:
        """Remove a registry hook."""
        with self._lock:
            if callback in self._hooks[event]:
                self._hooks[event].remove(callback)

    def _call_hooks(self, event: str, *args, **kwargs) -> None:
        """Call all hooks for an event."""
        for hook in self._hooks.get(event, []):
            try:
                hook(*args, **kwargs)
            except Exception as e:
                print(f"Hook error for event '{event}': {e}")

    def clear(self, category: str = None) -> None:
        """Clear registry entries."""
        with self._lock:
            if category:
                self._registry[category].clear()
                self._categories[category].clear()
            else:
                self._registry.clear()
                self._categories.clear()
                self._aliases.clear()
                self._factories.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            total_components = sum(len(cat) for cat in self._registry.values())
            category_counts = {cat: len(components)
                             for cat, components in self._registry.items()}

            return {
                "total_components": total_components,
                "total_categories": len(self._registry),
                "total_factories": len(self._factories),
                "total_aliases": len(self._aliases),
                "category_counts": category_counts
            }


# Global registry instance
registry = ComponentRegistry()


# Decorators for easy registration
def register_component(name: str,
                      category: str = "general",
                      factory_class: Type[BaseFactory] = None,
                      config_class: Type[BaseConfig] = None,
                      description: str = "",
                      tags: List[str] = None,
                      aliases: List[str] = None,
                      replace: bool = False):
    """
    Decorator for registering components.

    Usage:
        @register_component("my_encoder", category="encoders")
        class MyEncoder(BaseEncoder):
            pass
    """
    def decorator(cls: Type) -> Type:
        registry.register(
            name=name,
            component_class=cls,
            factory_class=factory_class,
            config_class=config_class,
            category=category,
            description=description,
            tags=tags,
            aliases=aliases,
            replace=replace
        )
        return cls
    return decorator


def register_encoder(name: str,
                    factory_class: Type[BaseFactory] = None,
                    config_class: Type[BaseConfig] = None,
                    description: str = "",
                    tags: List[str] = None,
                    aliases: List[str] = None):
    """Decorator specifically for encoders."""
    return register_component(
        name=name,
        category="encoders",
        factory_class=factory_class,
        config_class=config_class,
        description=description,
        tags=(tags or []) + ["encoder"],
        aliases=aliases
    )


def register_model(name: str,
                  factory_class: Type[BaseFactory] = None,
                  config_class: Type[BaseConfig] = None,
                  description: str = "",
                  tags: List[str] = None,
                  aliases: List[str] = None):
    """Decorator specifically for models."""
    return register_component(
        name=name,
        category="models",
        factory_class=factory_class,
        config_class=config_class,
        description=description,
        tags=(tags or []) + ["model"],
        aliases=aliases
    )


def register_processor(name: str,
                      factory_class: Type[BaseFactory] = None,
                      config_class: Type[BaseConfig] = None,
                      description: str = "",
                      tags: List[str] = None,
                      aliases: List[str] = None):
    """Decorator specifically for processors."""
    return register_component(
        name=name,
        category="processors",
        factory_class=factory_class,
        config_class=config_class,
        description=description,
        tags=(tags or []) + ["processor"],
        aliases=aliases
    )


def register_loss(name: str,
                 factory_class: Type[BaseFactory] = None,
                 config_class: Type[BaseConfig] = None,
                 description: str = "",
                 tags: List[str] = None,
                 aliases: List[str] = None):
    """Decorator specifically for loss functions."""
    return register_component(
        name=name,
        category="losses",
        factory_class=factory_class,
        config_class=config_class,
        description=description,
        tags=(tags or []) + ["loss"],
        aliases=aliases
    )