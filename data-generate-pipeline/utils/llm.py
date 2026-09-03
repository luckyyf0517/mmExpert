"""Shared OpenAI client and LLM call utilities.

All steps that need LLM access should use ``init_client()`` and
``call_llm_with_retry()`` from this module to keep client configuration
and retry logic consistent.
"""

from __future__ import annotations

import os
import time

import httpx
from openai import OpenAI


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def init_client() -> OpenAI:
    """Return a shared ``OpenAI`` client initialised from environment vars.

    Reads ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``) and
    ``OPENAI_API_KEY`` (required).
    """
    global _client
    if _client is not None:
        return _client

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    _client = OpenAI(
        base_url=base_url,
        api_key=os.environ["OPENAI_API_KEY"],
        http_client=httpx.Client(
            base_url=base_url,
            follow_redirects=True,
            timeout=30.0,
        ),
    )
    return _client


# ---------------------------------------------------------------------------
# Retryable error detection
# ---------------------------------------------------------------------------

_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Return *True* if *exc* is a transient error worth retrying."""
    if isinstance(exc, (httpx.RequestError, httpx.TimeoutException)):
        return True
    status = getattr(exc, "status_code", None)
    return status in _RETRYABLE_CODES  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Unified LLM call with retry
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
BACKOFF_FACTOR = 2.0


def call_llm_with_retry(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    max_tokens: int = 1000,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
    backoff_factor: float = BACKOFF_FACTOR,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> tuple[str, dict[str, int]]:
    """Call the LLM with exponential-backoff retry.

    Returns
    -------
    (response_text, usage_dict)
        ``usage_dict`` contains ``input_tokens`` and ``output_tokens``.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    delay = initial_delay
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            usage = completion.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            return (
                completion.choices[0].message.content or "",
                {"input_tokens": input_tokens, "output_tokens": output_tokens},
            )

        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < max_retries:
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise

    raise last_exc  # type: ignore[misc]
