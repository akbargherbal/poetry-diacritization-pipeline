import json
import logging
import os
import sys
import threading
import time
from collections import deque

import openai
from openai import OpenAI

from . import config

log = logging.getLogger("poetry_diacritization")


class SlidingWindowRateLimiter:
    """Thread-safe rate limiter, sliding window, no lock-sleep bottleneck."""

    def __init__(self, max_requests, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            sleep_time = 0
            with self.lock:
                now = time.time()
                while self.request_times and self.request_times[0] <= now - self.time_window:
                    self.request_times.popleft()
                if len(self.request_times) < self.max_requests:
                    self.request_times.append(now)
                    return
                sleep_time = self.time_window - (now - self.request_times[0]) + 0.1
            if sleep_time > 0:
                log.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)


# Per-provider connection details. Keeping this as one small table (rather
# than an if/else sprinkled through setup_client) is what makes adding a
# third provider later a one-entry change, not a re-read of this function.
_PROVIDER_CONNECTION = {
    config.PROVIDER_DEEPSEEK: {
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "label": "DeepSeek",
    },
    config.PROVIDER_NVIDIA: {
        "env_var": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "label": "NVIDIA",
    },
}


def setup_client(provider: str = None) -> OpenAI:
    """
    Build an authenticated OpenAI-SDK client for the given provider.
    `provider` defaults to config.MODEL_PROVIDER (i.e. whatever
    MODEL_PROVIDER was resolved to at import time) but can be overridden
    per call, same pattern as model/thinking_enabled elsewhere in this file.
    """
    provider = provider or config.MODEL_PROVIDER
    conn = _PROVIDER_CONNECTION.get(provider, _PROVIDER_CONNECTION[config.PROVIDER_DEEPSEEK])

    api_key = os.getenv(conn["env_var"])
    if not api_key:
        api_key = input(f"{conn['env_var']} not set. Paste your {conn['label']} API key: ").strip()

    client = OpenAI(api_key=api_key, base_url=conn["base_url"])

    try:
        client.models.list()
        log.info(f"{conn['label']} API authentication successful.")
    except openai.AuthenticationError:
        log.error("Authentication failed — check your API key.")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error contacting {conn['label']} API: {e}")
        sys.exit(1)

    return client


def build_user_prompt(verses: list) -> str:
    payload = json.dumps(verses, ensure_ascii=False)
    return f"{config.PROMPT_INSTRUCTION}\n\n{payload}"


def _extract_usage(response) -> dict | None:
    """
    Pull token counts out of the API response's `usage` block, for cost
    tracking (see pricing.py). Returns None — never raises — if the
    response has no usage info at all.

    DeepSeek's usage object splits input tokens into
    `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens` (which sum to
    `prompt_tokens`); if a deployment only sends one of the two, the other
    is derived rather than assumed to be zero.
    """
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return None

    prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
    completion_tokens = getattr(usage_obj, "completion_tokens", None)
    cache_hit_tokens = getattr(usage_obj, "prompt_cache_hit_tokens", None)
    cache_miss_tokens = getattr(usage_obj, "prompt_cache_miss_tokens", None)

    if cache_hit_tokens is None and prompt_tokens is not None and cache_miss_tokens is not None:
        cache_hit_tokens = prompt_tokens - cache_miss_tokens
    if cache_miss_tokens is None and prompt_tokens is not None and cache_hit_tokens is not None:
        cache_miss_tokens = prompt_tokens - cache_hit_tokens
    if cache_hit_tokens is None:
        cache_hit_tokens = 0
    if cache_miss_tokens is None:
        cache_miss_tokens = prompt_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
    }


def _build_extra_body(provider: str, thinking_enabled: bool, reasoning_effort: str) -> dict:
    """
    The thinking/reasoning-effort knobs are conceptually the same across
    providers (from the CLI's point of view: --thinking/--no-thinking,
    --reasoning-effort), but DeepSeek's native API and NVIDIA's NIM endpoint
    expect that intent wrapped in different `extra_body` shapes. This is the
    one place that difference lives, so call_llm itself doesn't need to
    know or care which provider it's talking to beyond passing this through.
    """
    if provider == config.PROVIDER_NVIDIA:
        # NVIDIA's OpenAI-compatible endpoint takes both knobs nested under
        # chat_template_kwargs. Unlike the DeepSeek branch, reasoning_effort
        # is sent unconditionally (harmless when thinking is off) to mirror
        # NVIDIA's own documented usage exactly.
        return {
            "chat_template_kwargs": {
                "thinking": bool(thinking_enabled),
                "reasoning_effort": reasoning_effort,
            }
        }
    # DeepSeek defaults thinking mode to ON — always send this explicitly
    # so "disabled" is a real, enforced setting, not an assumption.
    return {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}


def call_llm(
    client: OpenAI,
    verses: list,
    rate_limiter: SlidingWindowRateLimiter,
    model: str = None,
    thinking_enabled: bool = None,
    reasoning_effort: str = None,
    provider: str = None,
):
    """
    One API call for one batch of verses (a single poem's worth, <= BATCH_SIZE).
    No system prompt, by design. Returns
    (raw_text, reasoning_text_or_None, error_or_None, usage_dict_or_None).
    Never raises — network/API failure is reported back as an error string so
    the caller can mark those verse_ids for retry instead of crashing the pass.

    `usage_dict_or_None` is for cost tracking (see pricing.py) — it's
    always None on error, and best-effort (may still be None on success if
    the API response carried no usage block).

    model/thinking_enabled/reasoning_effort/provider default to config.py's
    settings but can be overridden per call (that's how the CLI's --model,
    --thinking/--no-thinking, and --reasoning-effort flags reach here).
    """
    provider = provider or config.MODEL_PROVIDER
    model = model or config.DEFAULT_MODEL
    if thinking_enabled is None:
        thinking_enabled = config.DEFAULT_THINKING_ENABLED
    reasoning_effort = reasoning_effort or config.DEFAULT_REASONING_EFFORT

    rate_limiter.acquire()
    prompt_text = build_user_prompt(verses)

    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        extra_body=_build_extra_body(provider, thinking_enabled, reasoning_effort),
    )
    # DeepSeek's native API takes reasoning_effort as a top-level param
    # (only meaningful/sent when thinking is on); NVIDIA takes it nested
    # inside extra_body's chat_template_kwargs instead (see
    # _build_extra_body) — never both, to avoid sending a param the
    # provider doesn't expect.
    if provider != config.PROVIDER_NVIDIA and thinking_enabled:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        message = response.choices[0].message
        # DeepSeek's SDK response exposes `reasoning_content`; NVIDIA's
        # exposes `reasoning`. Check both rather than assuming one.
        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        usage = _extract_usage(response)
        return content, reasoning, None, usage
    except Exception as e:
        log.error(f"LLM call failed for poem batch: {e}")
        return None, None, str(e), None
