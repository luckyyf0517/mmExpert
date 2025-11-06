"""
Dependency injection system for mmExpert framework.

This module provides a comprehensive dependency injection system that supports:
- Service registration and resolution
- Lifetime management (singleton, transient, scoped)
- Automatic dependency injection
- Circular dependency detection
- Type-based resolution
"""

import inspect
from typing import Any, Dict, List, Type, TypeVar, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from threading import Lock
import threading
import time
from abc import ABC, abstractmethod

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime options."""
    SINGLETON = "singleton"  # One instance for the entire application
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"        # One instance per scope


@dataclass
class ServiceDescriptor:
    """Describes a registered service."""
    interface: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: List[Type] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InjectionError(Exception):
    """Raised when dependency injection fails."""
    pass


class CircularDependencyError(InjectionError):
    """Raised when circular dependencies are detected."""
    pass


class DIContainer:
    """
    Dependency injection container.

    Manages service registration, resolution, and lifetime.
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = Lock()
        self._creation_stack: List[Type] = []
        self._scopes: Dict[str, Dict[Type, Any]] = {}

    def register_singleton(self,
                          interface: Type[T],
                          implementation: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None,
                          instance: Optional[T] = None) -> 'DIContainer':
        """
        Register a singleton service.

        Args:
            interface: Service interface type
            implementation: Implementation class (optional)
            factory: Factory function (optional)
            instance: Pre-created instance (optional)

        Returns:
            Self for method chaining
        """
        return self._register_service(
            interface=interface,
            implementation=implementation,
            factory=factory,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )

    def register_transient(self,
                          interface: Type[T],
                          implementation: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> 'DIContainer':
        """
        Register a transient service.

        Args:
            interface: Service interface type
            implementation: Implementation class (optional)
            factory: Factory function (optional)

        Returns:
            Self for method chaining
        """
        return self._register_service(
            interface=interface,
            implementation=implementation,
            factory=factory,
            instance=None,
            lifetime=ServiceLifetime.TRANSIENT
        )

    def register_scoped(self,
                       interface: Type[T],
                       implementation: Optional[Type[T]] = None,
                       factory: Optional[Callable[[], T]] = None) -> 'DIContainer':
        """
        Register a scoped service.

        Args:
            interface: Service interface type
            implementation: Implementation class (optional)
            factory: Factory function (optional)

        Returns:
            Self for method chaining
        """
        return self._register_service(
            interface=interface,
            implementation=implementation,
            factory=factory,
            instance=None,
            lifetime=ServiceLifetime.SCOPED
        )

    def register_instance(self, interface: Type[T], instance: T) -> 'DIContainer':
        """
        Register a pre-created instance as a singleton.

        Args:
            interface: Service interface type
            instance: Pre-created instance

        Returns:
            Self for method chaining
        """
        return self.register_singleton(interface, instance=instance)

    def _register_service(self,
                         interface: Type,
                         implementation: Optional[Type] = None,
                         factory: Optional[Callable] = None,
                         instance: Optional[Any] = None,
                         lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'DIContainer':
        """Internal service registration method."""
        with self._lock:
            # Validate registration
            if sum(x is not None for x in [implementation, factory, instance]) != 1:
                raise InjectionError(
                    f"Exactly one of implementation, factory, or instance must be provided for {interface}"
                )

            # Determine dependencies if implementation is provided
            dependencies = []
            if implementation:
                dependencies = self._get_dependencies(implementation)

            # Create service descriptor
            descriptor = ServiceDescriptor(
                interface=interface,
                implementation=implementation,
                factory=factory,
                instance=instance,
                lifetime=lifetime,
                dependencies=dependencies
            )

            self._services[interface] = descriptor

            # Store singleton instance if provided
            if lifetime == ServiceLifetime.SINGLETON and instance is not None:
                self._singletons[interface] = instance

        return self

    def resolve(self, interface: Type[T], scope: Optional[str] = None) -> T:
        """
        Resolve a service instance.

        Args:
            interface: Service interface type
            scope: Scope name for scoped services

        Returns:
            Service instance

        Raises:
            InjectionError: If service is not registered or resolution fails
        """
        with self._lock:
            if interface not in self._services:
                raise InjectionError(f"Service {interface} is not registered")

            descriptor = self._services[interface]

            # Check for circular dependencies
            if interface in self._creation_stack:
                cycle = " -> ".join(str(t) for t in self._creation_stack[::1]) + f" -> {interface}"
                raise CircularDependencyError(f"Circular dependency detected: {cycle}")

            try:
                # Return singleton if available
                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    if interface in self._singletons:
                        return self._singletons[interface]
                    if descriptor.instance is not None:
                        self._singletons[interface] = descriptor.instance
                        return descriptor.instance

                # Return scoped instance if available
                elif descriptor.lifetime == ServiceLifetime.SCOPED:
                    if scope is None:
                        raise InjectionError(f"Scope is required for scoped service {interface}")
                    if scope in self._scopes and interface in self._scopes[scope]:
                        return self._scopes[scope][interface]

                # Create new instance
                self._creation_stack.append(interface)
                instance = self._create_instance(descriptor, scope)
                self._creation_stack.pop()

                # Store instance based on lifetime
                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    self._singletons[interface] = instance
                elif descriptor.lifetime == ServiceLifetime.SCOPED:
                    if scope not in self._scopes:
                        self._scopes[scope] = {}
                    self._scopes[scope][interface] = instance

                return instance

            except Exception as e:
                self._creation_stack.clear()
                raise InjectionError(f"Failed to resolve {interface}: {e}")

    def _create_instance(self, descriptor: ServiceDescriptor, scope: Optional[str] = None) -> Any:
        """Create a new instance of a service."""
        # Use provided instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Use factory
        if descriptor.factory is not None:
            return descriptor.factory()

        # Use implementation class
        if descriptor.implementation is not None:
            return self._create_with_injection(descriptor.implementation, scope)

        raise InjectionError(f"No way to create instance for {descriptor.interface}")

    def _create_with_injection(self, cls: Type, scope: Optional[str] = None) -> Any:
        """Create instance with automatic dependency injection."""
        # Get constructor signature
        sig = inspect.signature(cls.__init__)

        # Prepare constructor arguments
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Check if parameter has type annotation
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation

                # Try to resolve dependency
                if param_type in self._services:
                    kwargs[param_name] = self.resolve(param_type, scope)
                elif param.default != inspect.Parameter.empty:
                    # Use default value if available
                    kwargs[param_name] = param.default
                else:
                    raise InjectionError(f"Cannot resolve dependency {param_type} for {cls}")

            elif param.default != inspect.Parameter.empty:
                # Use default value
                kwargs[param_name] = param.default

        return cls(**kwargs)

    def _get_dependencies(self, cls: Type) -> List[Type]:
        """Get dependency types from constructor signature."""
        dependencies = []
        sig = inspect.signature(cls.__init__)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            if param.annotation != inspect.Parameter.empty:
                dependencies.append(param.annotation)

        return dependencies

    def create_scope(self, scope_name: str) -> 'DIScope':
        """
        Create a new dependency injection scope.

        Args:
            scope_name: Name for the scope

        Returns:
            Scope object
        """
        return DIScope(self, scope_name)

    def clear_scope(self, scope_name: str) -> None:
        """Clear a scope and dispose its instances."""
        with self._lock:
            if scope_name in self._scopes:
                scope_instances = self._scopes[scope_name]
                # Call dispose method if available
                for instance in scope_instances.values():
                    if hasattr(instance, 'dispose') and callable(getattr(instance, 'dispose')):
                        try:
                            instance.dispose()
                        except Exception as e:
                            print(f"Error disposing instance: {e}")
                del self._scopes[scope_name]

    def get_registered_services(self) -> List[Type]:
        """Get list of registered service interfaces."""
        return list(self._services.keys())

    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered."""
        return interface in self._services


class DIScope:
    """
    Dependency injection scope.

    Manages scoped service instances for a specific context.
    """

    def __init__(self, container: DIContainer, scope_name: str):
        self.container = container
        self.scope_name = scope_name
        self._disposed = False

    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a service within this scope.

        Args:
            interface: Service interface type

        Returns:
            Service instance
        """
        if self._disposed:
            raise InjectionError("Scope has been disposed")

        return self.container.resolve(interface, self.scope_name)

    def dispose(self) -> None:
        """Dispose the scope and clean up resources."""
        if not self._disposed:
            self.container.clear_scope(self.scope_name)
            self._disposed = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.dispose()


class Injectable:
    """
    Mixin class for injectable services.

    Classes that inherit from this can use dependency injection
    for their constructor parameters.
    """

    def __init__(self, **kwargs):
        # Store injected dependencies
        self._injected_dependencies = kwargs


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global dependency injection container."""
    return _container


def injectable(lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
    """
    Decorator for making classes injectable.

    Args:
        lifetime: Service lifetime
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Register the class with itself as interface
        if lifetime == ServiceLifetime.SINGLETON:
            _container.register_singleton(cls, cls)
        elif lifetime == ServiceLifetime.SCOPED:
            _container.register_scoped(cls, cls)
        else:
            _container.register_transient(cls, cls)

        return cls
    return decorator


def inject(interface: Type[T], scope: Optional[str] = None) -> T:
    """
    Inject a service dependency.

    Args:
        interface: Service interface type
        scope: Scope for scoped services

    Returns:
        Service instance
    """
    return _container.resolve(interface, scope)


# Convenience functions
def register_singleton(interface: Type[T], implementation: Type[T] = None) -> None:
    """Register a singleton service."""
    _container.register_singleton(interface, implementation)


def register_transient(interface: Type[T], implementation: Type[T] = None) -> None:
    """Register a transient service."""
    _container.register_transient(interface, implementation)


def register_scoped(interface: Type[T], implementation: Type[T] = None) -> None:
    """Register a scoped service."""
    _container.register_scoped(interface, implementation)


def resolve(interface: Type[T]) -> T:
    """Resolve a service."""
    return _container.resolve(interface)


def create_scope(scope_name: str) -> DIScope:
    """Create a new scope."""
    return _container.create_scope(scope_name)