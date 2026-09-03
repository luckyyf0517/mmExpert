"""Summarize the quality of generated prompts with an OpenAI-compatible API."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

from ...logging_utils import get_console, get_logger
from utils.llm import init_client, call_llm_with_retry

logger = get_logger()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_prompts_from_directory(prompts_dir: str) -> dict[str, list[str]]:
    """Load all prompts from the prompts directory structure.

    Returns a dict mapping action directory names (e.g. "A00") to lists of
    prompt strings.
    """
    prompts_data: dict[str, list[str]] = {}
    prompts_path = Path(prompts_dir)

    if not prompts_path.exists():
        logger.error("Prompts directory not found: %s", prompts_dir)
        return prompts_data

    # Iterate through A00, A01, etc. directories
    for action_dir in sorted(prompts_path.glob("A[0-9][0-9]")):
        action_name = action_dir.name
        prompts_data[action_name] = []

        # Read all .txt files in this action directory
        for txt_file in sorted(action_dir.glob("*.txt")):
            try:
                content = txt_file.read_text(encoding="utf-8").strip()
                if content:
                    prompts_data[action_name].extend(content.split("\n"))
            except Exception as e:
                logger.error("Error reading %s: %s", txt_file, e)

    return prompts_data


def sample_prompts(
    prompts_data: dict[str, list[str]],
    samples_per_action: int = 10,
) -> dict[str, list[str]]:
    """Sample a subset of prompts from each action for analysis."""
    sampled_data: dict[str, list[str]] = {}

    for action_name, prompts in prompts_data.items():
        if prompts:
            sample_size = min(samples_per_action, len(prompts))
            sampled_prompts = random.sample(prompts, sample_size)
            sampled_data[action_name] = sampled_prompts

    return sampled_data


def create_analysis_prompt(
    prompts_data: dict[str, list[str]],
    config_requirements: dict[str, Any],
) -> str:
    """Create the analysis prompt for OpenAI."""
    prompt_parts = [
        "Please analyze the quality of motion description prompts generated for different actions.",
        "",
        "## CONFIGURATION REQUIREMENTS",
        "The original generation had these requirements:",
        f"- Model: {config_requirements.get('model', 'Unknown')}",
        "- Template requirements:",
    ]

    # Add template requirements
    template = config_requirements.get("prompt_template", [])
    if template:
        for i, req in enumerate(template, 1):
            req = req.replace('f"', "").replace("f'", "").strip()
            if req:
                prompt_parts.append(f"  {i}. {req}")

    prompt_parts.extend([
        "",
        "## TARGET ACTIONS",
        f"Total actions: {len(config_requirements.get('actions', []))}",
    ])

    actions = config_requirements.get("actions", [])
    for i, action in enumerate(actions, 1):
        prompt_parts.append(f"{i}. {action}")

    prompt_parts.extend([
        "",
        "## GENERATED PROMPTS SAMPLES",
        "Here are samples of the generated prompts for each action:",
        "",
    ])

    # Add sampled prompts for each action
    for action_name, prompts in prompts_data.items():
        if prompts:
            action_idx = int(action_name[1:])  # Extract number from "A00", "A01", etc.
            if action_idx < len(actions):
                original_action = actions[action_idx]
                prompt_parts.append(f"### {action_name} ({original_action})")
                prompt_parts.append(
                    f"Total prompts in this category: "
                    f"{len(prompts_data[action_name])} (showing {len(prompts)} samples)"
                )
                for i, prompt in enumerate(prompts, 1):
                    prompt_parts.append(f"{i}. {prompt}")
                prompt_parts.append("")

    prompt_parts.extend([
        "## ANALYSIS TASK",
        "Please provide a comprehensive analysis following this format:",
        "",
        "### 1. Overall Quality Assessment",
        "- Rate the overall prompt quality (1-10 scale)",
        "- Assess diversity and creativity",
        "- Evaluate adherence to requirements",
        "",
        "### 2. Requirement Compliance Analysis",
        "- Evaluate compliance with each template requirement",
        "- Identify common violations or issues",
        "- Highlight specific areas needing improvement",
        "",
        "### 3. Action-Specific Analysis",
        "- For each action, assess prompt quality and relevance",
        "- Identify which actions have better/worse prompts",
        "- Note any action-specific issues",
        "",
        "### 4. Common Issues and Patterns",
        "- List frequent problems found across prompts",
        "- Identify patterns in errors or inconsistencies",
        "- Note any systematic issues",
        "",
        "### 5. Recommendations",
        "- Specific suggestions for improving prompt generation",
        "- Template adjustments needed",
        "- Parameter changes (temperature, etc.)",
        "- Other optimization recommendations",
        "",
        "### 6. Summary Statistics",
        "- Total prompts analyzed",
        "- Average quality score",
        "- Compliance rate with requirements",
        "- Action-wise quality distribution",
        "",
        "Please be thorough and provide specific examples where possible.",
    ])

    return "\n".join(prompt_parts)


def analyze_prompts_with_ai(
    prompts_data: dict[str, list[str]],
    config_requirements: dict[str, Any],
    model: str = "gpt-4o-mini",
) -> str:
    """Send prompts to OpenAI for quality analysis."""
    console = get_console()

    console.print("[yellow]Creating analysis prompt...[/]")
    analysis_prompt = create_analysis_prompt(prompts_data, config_requirements)

    console.print("[yellow]Sending to OpenAI for analysis...[/]")
    try:
        client = init_client()
        system_prompt = (
            "You are an expert at analyzing text generation quality and "
            "providing detailed, constructive feedback. Focus on being "
            "thorough, specific, and actionable."
        )
        response_text, _usage = call_llm_with_retry(
            client,
            system_prompt,
            analysis_prompt,
            model=model,
            temperature=0.3,
            max_tokens=4000,
        )
        return response_text

    except Exception as e:
        logger.error("Error calling OpenAI API: %s", e)
        return f"Error: Failed to get analysis from OpenAI. {e}"


def create_summary_markdown(
    analysis_result: str,
    prompts_data: dict[str, list[str]],
    config_requirements: dict[str, Any],
    version: str,
) -> str:
    """Create a markdown summary document."""
    # Calculate statistics
    total_prompts = sum(len(prompts) for prompts in prompts_data.values())
    total_actions = len(prompts_data)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = [
        "# Prompt Quality Analysis Summary",
        "",
        f"**Version:** {version}",
        f"**Analysis Date:** {timestamp}",
        f"**Model Used:** {config_requirements.get('model', 'Unknown')}",
        "",
        "## Overview",
        "",
        f"- **Total Actions:** {total_actions}",
        f"- **Total Prompts Generated:** {total_prompts}",
        f"- **Average Prompts per Action:** {total_prompts // total_actions if total_actions > 0 else 0}",
        "",
        "## Configuration Requirements",
        "",
        "### Template Requirements",
    ]

    template = config_requirements.get("prompt_template", [])
    if template:
        for i, req in enumerate(template, 1):
            req = req.replace('f"', "").replace("f'", "").strip()
            if req:
                md_content.append(f"{i}. {req}")

    md_content.extend(["", "### Target Actions", ""])

    actions = config_requirements.get("actions", [])
    for i, action in enumerate(actions, 1):
        if prompts_data and i - 1 < len(prompts_data):
            action_key = f"A{i - 1:02d}"
            prompt_count = len(prompts_data.get(action_key, []))
            md_content.append(f"{i}. **{action}** ({prompt_count} prompts)")
        else:
            md_content.append(f"{i}. {action}")

    md_content.extend([
        "",
        "## Prompt Statistics",
        "",
        "| Action | Prompt Count | Status |",
        "|--------|-------------|--------|",
    ])

    for i, action in enumerate(actions):
        action_key = f"A{i:02d}"
        if action_key in prompts_data:
            count = len(prompts_data[action_key])
            status = "Generated" if count > 0 else "Missing"
        else:
            count = 0
            status = "Missing"
        md_content.append(f"| {action} | {count} | {status} |")

    md_content.extend([
        "",
        "## AI Quality Analysis",
        "",
        analysis_result,
        "",
        "---",
        f"*Generated automatically on {timestamp}*",
    ])

    return "\n".join(md_content)
