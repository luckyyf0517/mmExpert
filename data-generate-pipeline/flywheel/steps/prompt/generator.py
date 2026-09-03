"""Agentic Step 1: rewriter → planner → sub-planner → worker prompt generation.

Three-stage prompt generation pipeline:
1. Rewriter extracts action vocabulary from free-form task description.
2. Top planner designs strategic coverage directions (user approval).
3. Sub-planner per strategy decides quota and worker guidance.
4. Worker batches generate, deduplicate, and repair prompts.
"""

from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI

from ...config import FlywheelConfig, require_env
from ...logging_utils import get_console, get_logger, make_progress, step_panel
from ...path_manager import PathManager
from .display import print_config_summary, print_results_table
from utils.llm import init_client, call_llm_with_retry

logger = get_logger()
console = get_console()


@dataclass
class PlannerSubtask:
    """A planner-defined subtask with a target quota."""

    id: str
    title: str
    focus: str
    guidance: list[str]
    quota: int
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "focus": self.focus,
            "guidance": self.guidance,
            "quota": self.quota,
            "tags": self.tags,
        }


@dataclass
class RewriterResult:
    """Output of the action rewriter (Stage 0)."""

    original_prompt: str
    rewritten_prompt: str
    action_vocabulary: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "rewritten_prompt": self.rewritten_prompt,
            "action_vocabulary": self.action_vocabulary,
        }


@dataclass
class Strategy:
    """A strategic coverage direction from the top planner (Stage 1)."""

    id: str
    title: str
    description: str
    guidance: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "guidance": self.guidance,
        }


@dataclass
class BatchTask:
    """A single worker batch request."""

    batch_id: str
    subtask: PlannerSubtask
    target_count: int
    repair: bool = False
    is_seed: bool = False


@dataclass
class PromptGenResult:
    """Result of the prompt generation step."""

    total_prompts: int = 0
    final_prompt_count: int = 0
    existing_prompts: int = 0
    total_batches: int = 0
    repair_batches: int = 0
    rewritten_prompt: str = ""
    planned_strategies: int = 0
    planned_subtasks: int = 0
    duplicates_removed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    elapsed: float = 0.0
    batch_outputs: list[dict[str, Any]] = field(default_factory=list)


class Step1PromptGen:
    """Generate diverse motion description prompts via planner + workers."""

    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._config: FlywheelConfig | None = None
        self._result: PromptGenResult | None = None

    def run(
        self,
        version: str,
        config: FlywheelConfig,
        paths: PathManager,
        *,
        num_workers: int = 8,
        require_approval: bool = True,
    ) -> PromptGenResult:
        """Execute Step 1 with a three-stage planning workflow."""
        step_panel("Prompt Generation", subtitle=f"Version: {version}", step_num=1)
        require_env("OPENAI_API_KEY")

        if not config.tasks.strip():
            raise ValueError("`tasks` is empty in info.json.")
        if config.total_count <= 0:
            raise ValueError("`total_count` must be positive in info.json.")

        self._client = init_client()
        self._config = config
        self._result = PromptGenResult()
        result = self._result

        paths.prompts_dir.mkdir(parents=True, exist_ok=True)
        batches_dir = paths.step1_dir / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        existing_prompts = self._load_existing_prompts(paths.prompts_dir)
        result.existing_prompts = len(existing_prompts)

        print_config_summary(config)

        target_remaining = max(0, config.total_count - len(existing_prompts))
        if target_remaining == 0:
            result.final_prompt_count = len(existing_prompts)
            print_results_table(result, elapsed=0.0)
            return result

        batch_size = max(1, config.batch_size)

        # Stage 0: Action rewriter
        rewrite = self._rewrite_actions()
        result.rewritten_prompt = rewrite.rewritten_prompt
        console.print(
            f"  [green]Actions:[/] {', '.join(rewrite.action_vocabulary)}"
        )
        console.print(
            f"  [green]Rewritten:[/] {rewrite.rewritten_prompt[:80]}..."
        )

        # Stage 1: Top planner → strategic directions
        strategies = self._plan_strategies(rewrite)
        result.planned_strategies = len(strategies)
        console.print(
            f"  [green]Strategies:[/] [bold]{len(strategies)}[/]"
        )
        for s in strategies:
            console.print(f"    [dim]{s.id}[/] {s.title}")

        # User approval checkpoint
        if require_approval:
            plan_path = paths.step1_dir / "plan.json"
            self._save_json(
                plan_path,
                {
                    "version": version,
                    "tasks": config.tasks,
                    "total_count": config.total_count,
                    "remaining_count": target_remaining,
                    "rewrite": rewrite.to_dict(),
                    "strategies": [s.to_dict() for s in strategies],
                    "subtasks": None,  # filled after sub-planner
                },
            )
            console.print(
                "\n[yellow]Plan saved. Review strategies above.[/]"
            )
            try:
                input("  Press Enter to continue, Ctrl+C to abort... ")
            except KeyboardInterrupt:
                console.print("[red]Aborted by user.[/]")
                raise
            console.print("")

        # Stage 2: Sub-planner per strategy (concurrent)
        subtasks = self._expand_strategies(strategies, rewrite)
        subtasks = self._rebalance_quotas(subtasks, target_remaining)
        result.planned_subtasks = len(subtasks)
        console.print(
            f"  [green]Subtasks:[/] [bold]{len(subtasks)}[/] "
            f"(total quota: {sum(s.quota for s in subtasks)})"
        )

        plan_path = paths.step1_dir / "plan.json"
        self._save_json(
            plan_path,
            {
                "version": version,
                "tasks": config.tasks,
                "total_count": config.total_count,
                "remaining_count": target_remaining,
                "rewrite": rewrite.to_dict(),
                "strategies": [s.to_dict() for s in strategies],
                "subtasks": [s.to_dict() for s in subtasks],
            },
        )

        # Stage 3: Worker batches
        batch_tasks = self._create_batch_tasks(subtasks, batch_size)
        result.total_batches = len(batch_tasks)

        generated_prompts = self._run_batch_tasks(
            batch_tasks=batch_tasks,
            batches_dir=batches_dir,
            existing_prompts=existing_prompts,
            num_workers=num_workers,
        )

        unique_prompts = self._deduplicate_prompts(existing_prompts + generated_prompts)
        prompts_after_primary = unique_prompts
        generated_unique_count = max(0, len(prompts_after_primary) - len(existing_prompts))
        result.total_prompts = generated_unique_count
        result.duplicates_removed = len(existing_prompts) + len(generated_prompts) - len(prompts_after_primary)

        if len(prompts_after_primary) < config.total_count:
            repaired_prompts = self._repair_missing_prompts(
                current_prompts=prompts_after_primary,
                target_count=config.total_count,
                batch_size=batch_size,
                batches_dir=batches_dir,
                action_vocabulary=rewrite.action_vocabulary,
                subtasks=subtasks,
            )
            unique_prompts = self._deduplicate_prompts(prompts_after_primary + repaired_prompts)
            result.repair_batches = math.ceil(max(0, config.total_count - len(prompts_after_primary)) / batch_size)
            result.duplicates_removed += len(prompts_after_primary) + len(repaired_prompts) - len(unique_prompts)

        final_prompts = unique_prompts[: config.total_count]
        result.final_prompt_count = len(final_prompts)
        result.elapsed = time.time() - start_time

        self._write_prompt_files(final_prompts, paths.prompts_dir)
        self._save_json(
            paths.step1_dir / "prompts_metadata.json",
            {
                "tasks": config.tasks,
                "total_count": config.total_count,
                "final_prompt_count": len(final_prompts),
                "planner_model": config.planner_model,
                "worker_model": config.worker_model,
                "batch_size": batch_size,
                "batch_outputs": result.batch_outputs,
            },
        )

        print_results_table(result, result.elapsed)
        return result

    # ------------------------------------------------------------------
    # JSON retry wrapper
    # ------------------------------------------------------------------

    def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        temperature: float = 0.5,
        max_tokens: int = 2000,
        json_retries: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call LLM and parse JSON response, retrying on parse failure."""
        assert self._client is not None
        if json_retries < 1:
            json_retries = 1
        current_prompt = user_prompt
        for attempt in range(json_retries):
            response, usage = call_llm_with_retry(
                self._client,
                system_prompt,
                current_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            self._add_usage(usage)
            try:
                return self._parse_json_response(response)
            except ValueError as exc:
                if attempt < json_retries - 1:
                    logger.warning(
                        "JSON parse failed (attempt %d/%d): %s. Retrying...",
                        attempt + 1, json_retries, exc,
                    )
                    current_prompt = (
                        user_prompt
                        + "\n\nYour previous response was not valid JSON. "
                        f"Error: {exc}\n"
                        "Please return the same content as valid JSON only."
                    )
                else:
                    raise
        raise ValueError("unreachable")

    # ------------------------------------------------------------------
    # Stage 0: Action rewriter
    # ------------------------------------------------------------------

    def _rewrite_actions(self) -> RewriterResult:
        """Extract action vocabulary and rewrite task description."""
        assert self._config is not None

        system_prompt = (
            "You analyze a motion-prompt task description. "
            "Extract the action vocabulary and rewrite the description. "
            "Return valid JSON only."
        )
        user_prompt = (
            f"Original task description:\n{self._config.tasks}\n\n"
            "Return a JSON object with keys:\n"
            "- rewritten_prompt: a concise, canonical version of the task description\n"
            "- action_vocabulary: array of distinct action names mentioned or implied\n\n"
            "Rules:\n"
            "- Extract every distinct action, even if only implied.\n"
            "- Use canonical single-word forms (e.g. \"walk\" not \"walking around\").\n"
            "- Include composite actions if mentioned (e.g. \"walk-then-wave\").\n"
            "- The rewritten prompt should preserve the full intent but be more precise.\n"
            "- Output JSON only, no markdown."
        )

        raw = self._call_llm_json(
            system_prompt, user_prompt,
            model=self._config.worker_model,
            temperature=0.3,
            max_tokens=800,
        )
        vocab = raw.get("action_vocabulary", [])
        if isinstance(vocab, str):
            vocab = [vocab]
        return RewriterResult(
            original_prompt=self._config.tasks,
            rewritten_prompt=str(raw.get("rewritten_prompt", self._config.tasks)),
            action_vocabulary=[str(v).strip() for v in vocab if str(v).strip()],
        )

    def _task_scope_rules(self) -> list[str]:
        """Infer hard scope rules from the original task text."""
        assert self._config is not None

        task_text = self._config.tasks.lower()
        rules = [
            "Always prioritize the original user task over planner creativity or coverage expansion.",
            "Do not introduce actions, structures, or constraints outside the original user task.",
        ]

        single_only_markers = [
            "exactly one single action",
            "single action",
            "only these six actions",
            "do not generate combined",
            "do not generate combined, sequential, or multi-action captions",
            "multi-action",
            "sequential",
        ]
        if any(marker in task_text for marker in single_only_markers):
            rules.extend([
                "This task is single-action only.",
                "Do not generate, plan, suggest, or repair sequential, simultaneous, combined, chained, or multi-action captions.",
                "Every caption must describe exactly one action from the allowed action vocabulary.",
                "Do not extend beyond the listed action vocabulary.",
            ])

        return rules

    # ------------------------------------------------------------------
    # Stage 1: Top planner → strategies
    # ------------------------------------------------------------------

    def _plan_strategies(self, rewrite: RewriterResult) -> list[Strategy]:
        """Design strategic coverage directions."""
        assert self._config is not None

        system_prompt = (
            "You are a planner for a motion-prompt generation pipeline. "
            "Design diverse coverage strategies. Return valid JSON only."
        )
        user_prompt = (
            f"Original task: {rewrite.original_prompt}\n"
            f"Rewritten task: {rewrite.rewritten_prompt}\n"
            f"Action vocabulary: {rewrite.action_vocabulary}\n"
            f"Hard scope rules: {self._task_scope_rules()}\n"
            f"Existing constraints: {self._config.constraints or []}\n\n"
            "Design 5-10 strategic coverage directions for generating diverse "
            "motion description prompts. Strategies are types of motion scenarios, "
            "not per-action categories.\n\n"
            "Example in-scope strategy types:\n"
            "- Single-action variants (speed/direction/repetition)\n"
            "- Confusingly similar motions within the allowed action set\n"
            "- Context-rich scenarios that still describe exactly one action\n"
            "- Body-part emphasis and execution-style variations within scope\n\n"
            "Return JSON with keys:\n"
            "- task_understanding: string\n"
            "- strategies: array of objects\n\n"
            "Each strategy must have keys:\n"
            "- id: string (e.g. \"S01\")\n"
            "- title: short descriptive string\n"
            "- description: 1-2 sentences explaining what this strategy covers\n"
            "- guidance: array of actionable strings for downstream sub-planners\n\n"
            "Rules:\n"
            "- Produce 5-10 strategies.\n"
            "- Strategies must be mutually complementary, not overlapping.\n"
            "- All strategies must stay strictly within the original task and hard scope rules.\n"
            "- If the task is single-action only, every strategy must remain single-action only.\n"
            "- Do not add new actions beyond the provided action vocabulary unless the original task explicitly allows it.\n"
            "- Do NOT allocate quotas — that is decided later.\n"
            "- Output JSON only, no markdown."
        )

        raw = self._call_llm_json(
            system_prompt, user_prompt,
            model=self._config.planner_model,
            temperature=0.4,
            max_tokens=2000,
        )
        return self._parse_strategies(raw)

    @staticmethod
    def _parse_strategies(raw: dict[str, Any]) -> list[Strategy]:
        """Parse planner output into Strategy objects."""
        strategies: list[Strategy] = []
        for index, item in enumerate(raw.get("strategies", [])):
            guidance = item.get("guidance", [])
            if isinstance(guidance, str):
                guidance = [guidance]
            strategies.append(Strategy(
                id=str(item.get("id", f"S{index:02d}")),
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                guidance=[str(g).strip() for g in guidance if str(g).strip()],
            ))
        return strategies

    # ------------------------------------------------------------------
    # Stage 2: Sub-planner per strategy
    # ------------------------------------------------------------------

    def _expand_strategy(
        self,
        strategy: Strategy,
        rewrite: RewriterResult,
    ) -> list[PlannerSubtask]:
        """Sub-planner expands one strategy into concrete subtasks with quotas."""
        assert self._config is not None

        system_prompt = (
            "You decompose a prompt-generation strategy into concrete subtasks. "
            "You decide how many prompts each subtask should produce. "
            "Return valid JSON only."
        )
        user_prompt = (
            f"Original task: {rewrite.original_prompt}\n"
            f"Rewritten task: {rewrite.rewritten_prompt}\n"
            f"Action vocabulary: {rewrite.action_vocabulary}\n"
            f"Hard scope rules: {self._task_scope_rules()}\n"
            f"Global constraints: {self._config.constraints or []}\n\n"
            f"Strategy: {strategy.title}\n"
            f"Description: {strategy.description}\n"
            f"Strategy guidance: {strategy.guidance}\n\n"
            "Decompose this strategy into 5-15 concrete subtasks.\n"
            "Each subtask should cover a different combination of these dimensions:\n"
            "- action_structure: only structures allowed by the original task\n"
            "- variant: speed / direction / intensity / repetition\n"
            "- context: none / light / rich\n"
            "- style: short_annotation / natural_description\n\n"
            "Return JSON with keys:\n"
            "- subtasks: array of objects\n\n"
            "Each subtask must have keys:\n"
            "- id: string\n"
            "- focus: specific aspect this subtask covers\n"
            "- guidance: array of actionable strings for prompt writers\n"
            "- quota: integer, how many prompts this subtask should produce\n"
            "- tags: array of 2-4 coverage tags from the dimensions above\n"
            "  (e.g. [\"single\", \"fast\", \"light_context\"])\n\n"
            "Rules:\n"
            "- Decide quotas yourself based on how much diversity this focus needs.\n"
            "- Subtasks within a strategy should cover different dimension combinations.\n"
            "- Use the action vocabulary where appropriate.\n"
            "- All subtasks must obey the hard scope rules from the original task.\n"
            "- If the task is single-action only, every subtask must stay single-action only.\n"
            "- Output JSON only, no markdown."
        )

        raw = self._call_llm_json(
            system_prompt, user_prompt,
            model=self._config.worker_model,
            temperature=0.5,
            max_tokens=2000,
        )
        return self._parse_subtasks(raw, strategy)

    @staticmethod
    def _parse_subtasks(raw: dict[str, Any], strategy: Strategy) -> list[PlannerSubtask]:
        """Parse sub-planner output into PlannerSubtask objects."""
        subtasks: list[PlannerSubtask] = []
        for index, item in enumerate(raw.get("subtasks", [])):
            guidance = item.get("guidance", [])
            if isinstance(guidance, str):
                guidance = [guidance]
            tags = item.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            subtasks.append(PlannerSubtask(
                id=f"{strategy.id}-{item.get('id', f'T{index:02d}')}",
                title=f"{strategy.title}: {item.get('focus', '')}",
                focus=str(item.get("focus", "")).strip(),
                guidance=[str(g).strip() for g in guidance if str(g).strip()],
                quota=max(1, int(item.get("quota", 0) or 0)),
                tags=[str(t).strip() for t in tags if str(t).strip()],
            ))
        return subtasks

    def _expand_strategies(
        self,
        strategies: list[Strategy],
        rewrite: RewriterResult,
        *,
        num_workers: int = 8,
    ) -> list[PlannerSubtask]:
        """Run sub-planner concurrently for all strategies."""
        all_subtasks: list[PlannerSubtask] = []
        lock = Lock()
        progress = make_progress()
        max_workers = max(1, min(num_workers, len(strategies)))

        with progress:
            task_id = progress.add_task(
                "[cyan]Sub-planner[/]", total=len(strategies),
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._expand_strategy, strategy, rewrite,
                    ): strategy
                    for strategy in strategies
                }
                for future in as_completed(futures):
                    strategy = futures[future]
                    try:
                        subtasks = future.result()
                        with lock:
                            all_subtasks.extend(subtasks)
                    except Exception as exc:
                        logger.error(
                            "Sub-planner failed for %s: %s", strategy.id, exc,
                        )
                        raise
                    finally:
                        progress.update(task_id, advance=1)
        return all_subtasks

    @staticmethod
    def _rebalance_quotas(
        subtasks: list[PlannerSubtask],
        target_total: int,
    ) -> list[PlannerSubtask]:
        if not subtasks:
            return subtasks

        # If more subtasks than target, truncate and redistribute
        if len(subtasks) > target_total:
            truncated = sorted(subtasks, key=lambda s: s.quota, reverse=True)[:target_total]
            subtasks = truncated

        current_total = sum(item.quota for item in subtasks)
        if current_total == target_total:
            return subtasks

        adjusted = [PlannerSubtask(**subtask.to_dict()) for subtask in subtasks]
        index = 0
        while current_total < target_total:
            adjusted[index % len(adjusted)].quota += 1
            current_total += 1
            index += 1
        index = 0
        max_iterations = current_total * 2
        iterations = 0
        while current_total > target_total and iterations < max_iterations:
            candidate = adjusted[index % len(adjusted)]
            if candidate.quota > 1:
                candidate.quota -= 1
                current_total -= 1
            index += 1
            iterations += 1
        return adjusted

    @staticmethod
    def _create_batch_tasks(subtasks: list[PlannerSubtask], batch_size: int) -> list[BatchTask]:
        batch_tasks: list[BatchTask] = []
        batch_index = 0
        for subtask in subtasks:
            remaining = subtask.quota
            first = True
            while remaining > 0:
                current = min(batch_size, remaining)
                batch_tasks.append(
                    BatchTask(
                        batch_id=f"batch_{batch_index:03d}",
                        subtask=subtask,
                        target_count=current,
                        is_seed=first,
                    )
                )
                first = False
                batch_index += 1
                remaining -= current
        return batch_tasks

    def _run_batch_tasks(
        self,
        *,
        batch_tasks: list[BatchTask],
        batches_dir: Path,
        existing_prompts: list[str],
        num_workers: int,
    ) -> list[str]:
        if not batch_tasks:
            return []

        generated_prompts: list[str] = []
        lock = Lock()
        progress = make_progress()

        seed_batches = [b for b in batch_tasks if b.is_seed]
        expansion_batches = [b for b in batch_tasks if not b.is_seed]

        with progress:
            task_id = progress.add_task("[cyan]Worker batches[/]", total=len(batch_tasks))

            # Phase 1: Run all seed batches concurrently, collect seeds per subtask
            seeds_by_subtask: dict[str, list[str]] = {}
            seed_lock = Lock()
            max_workers = max(1, min(num_workers, len(seed_batches))) if seed_batches else 1

            if seed_batches:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._generate_batch,
                            bt, existing_prompts, batches_dir, None,
                        ): bt
                        for bt in seed_batches
                    }
                    for future in as_completed(futures):
                        bt = futures[future]
                        try:
                            prompts, batch_record = future.result()
                            with lock:
                                generated_prompts.extend(prompts)
                                assert self._result is not None
                                self._result.batch_outputs.append(batch_record)
                            with seed_lock:
                                seeds_by_subtask.setdefault(bt.subtask.id, []).extend(prompts)
                        except Exception as exc:
                            logger.error("Seed batch %s failed: %s", bt.batch_id, exc)
                            raise
                        finally:
                            progress.update(task_id, advance=1)

            # Phase 2: Run expansion batches with seeds
            max_workers = max(1, min(num_workers, len(expansion_batches))) if expansion_batches else 1

            if expansion_batches:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._generate_batch,
                            bt,
                            existing_prompts + generated_prompts,
                            batches_dir,
                            seeds_by_subtask.get(bt.subtask.id),
                        ): bt
                        for bt in expansion_batches
                    }
                    for future in as_completed(futures):
                        bt = futures[future]
                        try:
                            prompts, batch_record = future.result()
                            with lock:
                                generated_prompts.extend(prompts)
                                assert self._result is not None
                                self._result.batch_outputs.append(batch_record)
                        except Exception as exc:
                            logger.error("Expansion batch %s failed: %s", bt.batch_id, exc)
                            raise
                        finally:
                            progress.update(task_id, advance=1)

        return generated_prompts

    def _generate_batch(
        self,
        batch_task: BatchTask,
        existing_prompts: list[str],
        batches_dir: Path,
        seed_prompts: list[str] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        assert self._client is not None
        assert self._config is not None

        system_prompt = (
            "You generate concise, action-focused human motion descriptions. "
            "Return valid JSON only."
        )
        prompt_template_text = "\n".join(self._config.prompt_template)
        existing_sample = existing_prompts[-20:]

        seed_section = ""
        if seed_prompts:
            seed_sample = seed_prompts[:10]
            seed_section = (
                "\nSeed prompts from this subtask (use as direction reference, do NOT repeat them):\n"
                + "\n".join(f"- {s}" for s in seed_sample)
                + "\n"
            )

        user_prompt = (
            f"Global task scope:\n{self._config.tasks}\n\n"
            f"Hard scope rules:\n{self._task_scope_rules()}\n\n"
            f"Subtask title: {batch_task.subtask.title}\n"
            f"Subtask focus: {batch_task.subtask.focus}\n"
            f"Subtask guidance: {batch_task.subtask.guidance}\n"
            f"Global constraints: {self._config.constraints or []}\n"
            f"Prompt writing template guidance:\n{prompt_template_text}\n"
            f"{seed_section}\n"
            f"Generate exactly {batch_task.target_count} distinct motion descriptions.\n"
            "Each description must be a single sentence.\n"
            "Descriptions must stay within the task scope while maximizing diversity.\n"
            "Style rules (IMPORTANT):\n"
            "- Focus on the motion itself: body parts, direction, speed, repetition.\n"
            "- Avoid emotional qualifiers (enthusiastically, joyfully, cheerfully, etc.).\n"
            "- Avoid scenic or atmospheric filler (sunlit park, bustling market, golden glow, etc.).\n"
            "- Avoid repetitive sentence patterns like 'With a ..., [subject] [verb]'.\n"
            "- Keep descriptions concise and factual, like a motion-capture annotation.\n"
            "- Follow the original task scope exactly; do not add extra actions or multi-action structure.\n"
            "Avoid generic paraphrases and avoid repeating the same motion pattern.\n"
            f"Avoid semantic duplicates of these existing examples: {existing_sample}\n\n"
            "Return JSON with keys:\n"
            "- batch_id: string\n"
            "- prompts: string[]\n"
            "Output JSON only."
        )

        payload = self._call_llm_json(
            system_prompt,
            user_prompt,
            model=self._config.worker_model,
            temperature=0.8,
            max_tokens=900,
            frequency_penalty=0.6,
            presence_penalty=0.6,
        )
        prompts = payload.get("prompts", [])
        cleaned = self._clean_prompts(prompts)

        batch_record = {
            "batch_id": batch_task.batch_id,
            "subtask": batch_task.subtask.to_dict(),
            "target_count": batch_task.target_count,
            "repair": batch_task.repair,
            "generated_count": len(cleaned),
            "prompts": cleaned,
        }
        self._save_json(batches_dir / f"{batch_task.batch_id}.json", batch_record)
        return cleaned, batch_record

    def _repair_missing_prompts(
        self,
        *,
        current_prompts: list[str],
        target_count: int,
        batch_size: int,
        batches_dir: Path,
        action_vocabulary: list[str],
        subtasks: list[PlannerSubtask],
    ) -> list[str]:
        assert self._config is not None

        weak_actions = self._find_weak_actions(current_prompts, action_vocabulary)
        weak_tags = self._find_weak_tags(current_prompts, subtasks)

        repaired: list[str] = []
        repair_index = 0

        # Build targeted repair subtasks from weak buckets
        targeted_subtasks = self._build_repair_subtasks(
            weak_actions=weak_actions,
            weak_tags=weak_tags,
            total_gap=target_count - len(current_prompts),
            batch_size=batch_size,
        )

        # Run targeted repair batches first
        for repair_subtask in targeted_subtasks:
            remaining = target_count - len(current_prompts) - len(repaired)
            if remaining <= 0:
                break
            actual_count = min(repair_subtask.quota, remaining)
            repair_subtask.quota = actual_count

            batch_task = BatchTask(
                batch_id=f"repair_{repair_index:03d}",
                subtask=repair_subtask,
                target_count=actual_count,
                repair=True,
            )
            prompts, batch_record = self._generate_batch(
                batch_task,
                current_prompts + repaired,
                batches_dir,
            )
            repaired.extend(prompts)
            assert self._result is not None
            self._result.batch_outputs.append(batch_record)
            repair_index += 1

        # Fallback: if still short, generate generic repair batches
        empty_streak = 0
        while len(current_prompts) + len(repaired) < target_count:
            remaining = target_count - len(current_prompts) - len(repaired)
            fallback_focus = "Generate diverse additional prompts to reach target count."
            fallback_guidance = [
                "prioritize underrepresented motion variants",
                "avoid paraphrases of existing prompts",
                "keep prompts realistic and specific",
            ]
            if weak_actions:
                fallback_focus = (
                    f"Focus on underrepresented actions: {', '.join(weak_actions[:3])}. "
                    + fallback_focus
                )
            repair_subtask = PlannerSubtask(
                id=f"R{repair_index:02d}",
                title="Fallback repair",
                focus=fallback_focus,
                guidance=fallback_guidance,
                quota=min(batch_size, remaining),
                tags=weak_tags[:3] if weak_tags else [],
            )
            batch_task = BatchTask(
                batch_id=f"repair_{repair_index:03d}",
                subtask=repair_subtask,
                target_count=min(batch_size, remaining),
                repair=True,
            )
            prompts, batch_record = self._generate_batch(
                batch_task,
                current_prompts + repaired,
                batches_dir,
            )
            if not prompts:
                empty_streak += 1
                if empty_streak >= 3:
                    logger.warning(
                        "Repair returned empty %d times consecutively, stopping. "
                        "Final count: %d/%d",
                        empty_streak,
                        len(current_prompts) + len(repaired),
                        target_count,
                    )
                    break
            else:
                empty_streak = 0
            repaired.extend(prompts)
            assert self._result is not None
            self._result.batch_outputs.append(batch_record)
            repair_index += 1
        return repaired

    @staticmethod
    def _find_weak_actions(
        current_prompts: list[str],
        action_vocabulary: list[str],
        threshold: float = 0.5,
    ) -> list[str]:
        """Find actions whose frequency is below threshold * average."""
        if not action_vocabulary or not current_prompts:
            return []
        lowered_all = " ".join(p.lower() for p in current_prompts)
        counts = {action: lowered_all.count(action.lower()) for action in action_vocabulary}
        total = sum(counts.values())
        if total == 0:
            return list(action_vocabulary)
        avg = total / len(action_vocabulary)
        return [action for action, count in counts.items() if count < avg * threshold]

    @staticmethod
    def _find_weak_tags(
        current_prompts: list[str],
        subtasks: list[PlannerSubtask],
    ) -> list[str]:
        """Find tags from subtasks that are underrepresented in actual output."""
        if not subtasks:
            return []
        all_tags: set[str] = set()
        for subtask in subtasks:
            all_tags.update(subtask.tags)
        if not all_tags:
            return []
        lowered_all = " ".join(p.lower() for p in current_prompts)
        tag_counts: dict[str, int] = {}
        for tag in all_tags:
            tag_counts[tag] = lowered_all.count(tag.lower())
        total = sum(tag_counts.values())
        if total == 0:
            return list(all_tags)
        avg = total / len(tag_counts)
        return [tag for tag, count in tag_counts.items() if count < avg * 0.5]

    @staticmethod
    def _build_repair_subtasks(
        weak_actions: list[str],
        weak_tags: list[str],
        total_gap: int,
        batch_size: int,
    ) -> list[PlannerSubtask]:
        """Build targeted repair subtasks for weak buckets."""
        repair_subtasks: list[PlannerSubtask] = []

        # One targeted subtask per weak action
        for idx, action in enumerate(weak_actions):
            quota = min(batch_size, max(5, total_gap // max(1, len(weak_actions))))
            repair_subtasks.append(PlannerSubtask(
                id=f"RA{idx:02d}",
                title=f"Repair: {action} variants",
                focus=f"Generate diverse prompts featuring '{action}' with varied speed, direction, and context.",
                guidance=[
                    f"focus on the action '{action}'",
                    "vary speed, direction, and repetition",
                    "stay strictly within the original task scope",
                    "keep every caption to exactly one allowed action",
                    "avoid paraphrases of existing prompts",
                ],
                quota=quota,
                tags=[action, "repair"],
            ))

        # One targeted subtask for weak tags (if any and different from actions)
        if weak_tags:
            relevant_tags = [t for t in weak_tags if t not in weak_actions]
            if relevant_tags:
                quota = min(batch_size, max(5, total_gap // 3))
                repair_subtasks.append(PlannerSubtask(
                    id="RT00",
                    title=f"Repair: {', '.join(relevant_tags[:3])} coverage",
                    focus=f"Generate prompts emphasizing: {', '.join(relevant_tags[:3])}.",
                    guidance=[
                        f"emphasize these aspects: {', '.join(relevant_tags[:3])}",
                        "avoid repeating existing patterns",
                    ],
                    quota=quota,
                    tags=relevant_tags[:3] + ["repair"],
                ))

        return repair_subtasks

    @staticmethod
    def _clean_prompts(prompts: Any) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        cleaned: list[str] = []
        for item in prompts or []:
            text = str(item).strip()
            if not text:
                continue
            text = re.sub(r"^\d+[\.\)]\s*", "", text)
            text = " ".join(text.split())
            if text:
                cleaned.append(text)
        return cleaned

    @staticmethod
    def _deduplicate_prompts(prompts: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for prompt in prompts:
            key = Step1PromptGen._normalize_prompt_key(prompt)
            if key in seen:
                continue
            seen.add(key)
            unique.append(prompt)
        return unique

    @staticmethod
    def _normalize_prompt_key(prompt: str) -> str:
        lowered = prompt.lower().strip()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered

    @staticmethod
    def _parse_json_response(response: str) -> dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        # Strip control characters that LLM sometimes injects inside strings
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response must be an object.")
        return parsed

    @staticmethod
    def _save_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    @staticmethod
    def _load_existing_prompts(prompts_dir: Path) -> list[str]:
        prompts: list[str] = []
        if not prompts_dir.exists():
            return prompts
        for prompt_file in sorted(prompts_dir.glob("*.txt")):
            content = prompt_file.read_text(encoding="utf-8").strip()
            if content:
                prompts.extend(line.strip() for line in content.splitlines() if line.strip())
        return prompts

    @staticmethod
    def _write_prompt_files(prompts: list[str], prompts_dir: Path, file_capacity: int = 500) -> None:
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for existing_file in prompts_dir.glob("*.txt"):
            existing_file.unlink()
        for index in range(0, len(prompts), file_capacity):
            chunk = prompts[index:index + file_capacity]
            file_index = index // file_capacity
            target_file = prompts_dir / f"{file_index:04d}.txt"
            with open(target_file, "w", encoding="utf-8") as f:
                for prompt in chunk:
                    f.write(prompt + "\n")

    def _add_usage(self, usage: dict[str, int]) -> None:
        assert self._config is not None
        assert self._result is not None
        self._result.total_input_tokens += usage["input_tokens"]
        self._result.total_output_tokens += usage["output_tokens"]
        input_price = self._config.data.get("price_per_1000_input_tokens", 0.0)
        output_price = self._config.data.get("price_per_1000_output_tokens", 0.0)
        self._result.total_cost += (
            usage["input_tokens"] / 1000 * input_price
            + usage["output_tokens"] / 1000 * output_price
        )
