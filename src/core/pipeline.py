"""
Pipeline system for mmExpert framework.

This module provides a unified processing pipeline system that supports:
- Modular pipeline construction
- Sequential and parallel processing
- Conditional execution
- Pipeline composition and nesting
- Progress tracking and monitoring
"""

import time
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

from .base import BasePipeline, BaseProcessor, ModalityData, ModalityType
from .injection import injectable, ServiceLifetime, inject


T = TypeVar('T')


class ExecutionMode(Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class PipelineContext:
    """Context for pipeline execution."""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    execution_stats: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in context."""
        with self._lock:
            self.data[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """Update context with multiple values."""
        with self._lock:
            self.data.update(data)

    def get_step_result(self, step_name: str) -> Any:
        """Get result from a specific step."""
        return self.step_results.get(step_name)

    def set_step_result(self, step_name: str, result: Any) -> None:
        """Set result for a specific step."""
        with self._lock:
            self.step_results[step_name] = result

    def add_stat(self, key: str, value: Any) -> None:
        """Add execution statistic."""
        with self._lock:
            self.execution_stats[key] = value


@dataclass
class PipelineStep:
    """Single step in a pipeline."""
    name: str
    processor: BaseProcessor
    condition: Optional[Callable[[PipelineContext], bool]] = None
    enabled: bool = True
    parallel: bool = False
    max_workers: Optional[int] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    retry_delay: float = 0.0

    def should_execute(self, context: PipelineContext) -> bool:
        """Check if step should be executed based on condition."""
        if not self.enabled:
            return False
        if self.condition is None:
            return True
        return self.condition(context)


class PipelineExecutor:
    """Executes pipeline steps with various strategies."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def execute_sequential(self,
                          steps: List[PipelineStep],
                          context: PipelineContext) -> None:
        """Execute steps sequentially."""
        for step in steps:
            if step.should_execute(context):
                self._execute_step_with_retry(step, context)

    def execute_parallel(self,
                        steps: List[PipelineStep],
                        context: PipelineContext) -> None:
        """Execute steps in parallel."""
        # Filter steps that should execute
        executable_steps = [step for step in steps if step.should_execute(context)]

        if not executable_steps:
            return

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self._get_max_workers(executable_steps)) as executor:
            # Submit all tasks
            future_to_step = {
                executor.submit(self._execute_step, step, context): step
                for step in executable_steps
            }

            # Collect results
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    future.result(timeout=step.timeout)
                except Exception as e:
                    self.logger.error(f"Error in parallel step '{step.name}': {e}")
                    raise

    def execute_conditional(self,
                           steps: List[PipelineStep],
                           context: PipelineContext) -> None:
        """Execute steps with conditional logic."""
        for step in steps:
            if step.should_execute(context):
                self._execute_step_with_retry(step, context)
            else:
                self.logger.debug(f"Skipping conditional step '{step.name}'")

    def _execute_step_with_retry(self,
                                step: PipelineStep,
                                context: PipelineContext) -> Any:
        """Execute a step with retry logic."""
        last_exception = None

        for attempt in range(step.retry_count + 1):
            try:
                return self._execute_step(step, context)
            except Exception as e:
                last_exception = e
                if attempt < step.retry_count:
                    self.logger.warning(
                        f"Step '{step.name}' failed (attempt {attempt + 1}/{step.retry_count + 1}): {e}"
                    )
                    if step.retry_delay > 0:
                        time.sleep(step.retry_delay)
                else:
                    self.logger.error(f"Step '{step.name}' failed after {step.retry_count + 1} attempts: {e}")

        raise last_exception

    def _execute_step(self, step: PipelineStep, context: PipelineContext) -> Any:
        """Execute a single step."""
        start_time = time.time()

        try:
            self.logger.debug(f"Executing step '{step.name}'")

            # Get input data from context
            input_data = context.get(step.name, context.data)

            # Execute processor
            result = step.processor.process(input_data, context=context)

            # Store result
            context.set_step_result(step.name, result)

            # Record execution time
            execution_time = time.time() - start_time
            context.add_stat(f"{step.name}_execution_time", execution_time)

            self.logger.debug(f"Step '{step.name}' completed in {execution_time:.3f}s")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            context.add_stat(f"{step.name}_error", str(e))
            context.add_stat(f"{step.name}_execution_time", execution_time)
            raise

    def _get_max_workers(self, steps: List[PipelineStep]) -> int:
        """Determine maximum number of workers for parallel execution."""
        if any(step.max_workers for step in steps):
            # Use the maximum specified workers
            return max((step.max_workers or 1) for step in steps)
        else:
            # Use number of CPU cores
            import multiprocessing
            return multiprocessing.cpu_count()


class BasePipelineImpl(BasePipeline):
    """Base implementation of pipeline with common functionality."""

    def __init__(self,
                 name: str,
                 model=None,
                 execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
                 **kwargs):
        super().__init__(name, model, **kwargs)
        self.execution_mode = execution_mode
        self._steps: List[PipelineStep] = []
        self._executor = PipelineExecutor()
        self.logger = logging.getLogger(f"{__name__}.{name}")

    def add_step(self,
                 name: str,
                 processor: BaseProcessor,
                 condition: Optional[Callable[[PipelineContext], bool]] = None,
                 enabled: bool = True,
                 parallel: bool = False,
                 **kwargs) -> 'BasePipelineImpl':
        """
        Add a step to the pipeline.

        Args:
            name: Step name
            processor: Processor to execute
            condition: Condition function for execution
            enabled: Whether step is enabled
            parallel: Whether to execute in parallel
            **kwargs: Additional step parameters

        Returns:
            Self for method chaining
        """
        step = PipelineStep(
            name=name,
            processor=processor,
            condition=condition,
            enabled=enabled,
            parallel=parallel,
            **kwargs
        )
        self._steps.append(step)
        return self

    def add_condition(self, condition: Callable[[PipelineContext], bool]) -> 'BasePipelineImpl':
        """Add a condition to the last added step."""
        if self._steps:
            self._steps[-1].condition = condition
        else:
            raise ValueError("No steps added yet")
        return self

    def process(self, data: Any, **kwargs) -> Any:
        """Process data through the pipeline."""
        # Create execution context
        context = PipelineContext()

        # Initialize context with input data
        if isinstance(data, dict):
            context.update(data)
        else:
            context.set("input", data)

        # Add pipeline metadata
        context.metadata.update({
            "pipeline_name": self.name,
            "execution_mode": self.execution_mode.value,
            "step_count": len(self._steps)
        })

        # Execute pipeline
        start_time = time.time()

        try:
            self._execute_pipeline(context)

            # Record total execution time
            total_time = time.time() - start_time
            context.add_stat("total_execution_time", total_time)

            self.logger.info(f"Pipeline '{self.name}' completed in {total_time:.3f}s")

            return context.step_results

        except Exception as e:
            total_time = time.time() - start_time
            context.add_stat("pipeline_error", str(e))
            context.add_stat("total_execution_time", total_time)
            self.logger.error(f"Pipeline '{self.name}' failed after {total_time:.3f}s: {e}")
            raise

    def _execute_pipeline(self, context: PipelineContext) -> None:
        """Execute the pipeline with the configured mode."""
        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            self._executor.execute_sequential(self._steps, context)
        elif self.execution_mode == ExecutionMode.PARALLEL:
            self._executor.execute_parallel(self._steps, context)
        elif self.execution_mode == ExecutionMode.CONDITIONAL:
            self._executor.execute_conditional(self._steps, context)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")

    def get_step_names(self) -> List[str]:
        """Get list of step names."""
        return [step.name for step in self._steps]

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        for step in self._steps:
            if step.name == name:
                return step
        return None

    def remove_step(self, name: str) -> 'BasePipelineImpl':
        """Remove a step by name."""
        self._steps = [step for step in self._steps if step.name != name]
        return self

    def clear_steps(self) -> 'BasePipelineImpl':
        """Clear all steps."""
        self._steps.clear()
        return self


class ProcessingPipeline(BasePipelineImpl):
    """Standard pipeline for data processing."""

    def __init__(self, name: str, model=None, **kwargs):
        super().__init__(name, model, **kwargs)

    def preprocess(self, processor: BaseProcessor, name: str = "preprocess") -> 'ProcessingPipeline':
        """Add a preprocessing step."""
        return self.add_step(name, processor)

    def encode(self, processor: BaseProcessor, name: str = "encode") -> 'ProcessingPipeline':
        """Add an encoding step."""
        return self.add_step(name, processor)

    def postprocess(self, processor: BaseProcessor, name: str = "postprocess") -> 'ProcessingPipeline':
        """Add a postprocessing step."""
        return self.add_step(name, processor)

    def evaluate(self, processor: BaseProcessor, name: str = "evaluate") -> 'ProcessingPipeline':
        """Add an evaluation step."""
        return self.add_step(name, processor)


class ModelPipeline(BasePipelineImpl):
    """Pipeline specifically for model inference and evaluation."""

    def __init__(self, name: str, model, **kwargs):
        super().__init__(name, model, **kwargs)

        # Add model encoding step if model is provided
        if model:
            self.add_step("model_encode", ModelEncodingProcessor(model))

    def data_preparation(self, processor: BaseProcessor) -> 'ModelPipeline':
        """Add data preparation step."""
        return self.add_step("data_preparation", processor)

    def feature_extraction(self, processor: BaseProcessor) -> 'ModelPipeline':
        """Add feature extraction step."""
        return self.add_step("feature_extraction", processor)

    def similarity_computation(self, processor: BaseProcessor) -> 'ModelPipeline':
        """Add similarity computation step."""
        return self.add_step("similarity_computation", processor)

    def loss_computation(self, processor: BaseProcessor) -> 'ModelPipeline':
        """Add loss computation step."""
        return self.add_step("loss_computation", processor)


class ModelEncodingProcessor(BaseProcessor):
    """Processor for model encoding operations."""

    def __init__(self, model):
        self.model = model

    def process(self, data: Any, context: PipelineContext = None, **kwargs) -> Any:
        """Process data using model encoding."""
        if hasattr(self.model, 'encode'):
            return self.model.encode(data, **kwargs)
        elif hasattr(self.model, 'forward'):
            return self.model.forward(data, **kwargs)
        else:
            raise ValueError("Model does not have encode or forward method")


class ConditionalProcessor(BaseProcessor):
    """Processor that applies different processing based on conditions."""

    def __init__(self, processors: Dict[Any, BaseProcessor], default_processor: BaseProcessor = None):
        self.processors = processors
        self.default_processor = default_processor

    def process(self, data: Any, context: PipelineContext = None, **kwargs) -> Any:
        """Process data using conditional logic."""
        condition = kwargs.get("condition")

        if condition in self.processors:
            return self.processors[condition].process(data, context=context, **kwargs)
        elif self.default_processor:
            return self.default_processor.process(data, context=context, **kwargs)
        else:
            raise ValueError(f"No processor found for condition: {condition}")


# Pipeline builder for fluent API
class PipelineBuilder:
    """Builder for creating pipelines with fluent API."""

    def __init__(self, name: str):
        self.pipeline = ProcessingPipeline(name)

    def sequential(self) -> 'PipelineBuilder':
        """Set execution mode to sequential."""
        self.pipeline.execution_mode = ExecutionMode.SEQUENTIAL
        return self

    def parallel(self) -> 'PipelineBuilder':
        """Set execution mode to parallel."""
        self.pipeline.execution_mode = ExecutionMode.PARALLEL
        return self

    def conditional(self) -> 'PipelineBuilder':
        """Set execution mode to conditional."""
        self.pipeline.execution_mode = ExecutionMode.CONDITIONAL
        return self

    def step(self, name: str, processor: BaseProcessor, **kwargs) -> 'PipelineBuilder':
        """Add a step to the pipeline."""
        self.pipeline.add_step(name, processor, **kwargs)
        return self

    def build(self) -> BasePipelineImpl:
        """Build and return the pipeline."""
        return self.pipeline


# Convenience functions
def create_pipeline(name: str, execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> PipelineBuilder:
    """Create a new pipeline builder."""
    return PipelineBuilder(name).with_mode(execution_mode)


def sequential_pipeline(name: str) -> PipelineBuilder:
    """Create a sequential pipeline builder."""
    return create_pipeline(name, ExecutionMode.SEQUENTIAL)


def parallel_pipeline(name: str) -> PipelineBuilder:
    """Create a parallel pipeline builder."""
    return create_pipeline(name, ExecutionMode.PARALLEL)


def conditional_pipeline(name: str) -> PipelineBuilder:
    """Create a conditional pipeline builder."""
    return create_pipeline(name, ExecutionMode.CONDITIONAL)