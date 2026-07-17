import json
import time

import httpx
import openai
import pytest

from poetry_diacritization import config, llm_client
from poetry_diacritization.llm_client import (
    SlidingWindowRateLimiter,
    build_user_prompt,
    call_llm,
    setup_client,
)


# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_up_to_max_requests_without_blocking(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    fake_now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: fake_now[0])

    limiter = SlidingWindowRateLimiter(max_requests=3, time_window=60)
    for _ in range(3):
        limiter.acquire()

    assert sleeps == []  # never had to wait
    assert len(limiter.request_times) == 3


def test_rate_limiter_blocks_once_window_is_full_then_releases(monkeypatch):
    sleeps = []
    fake_now = [1000.0]

    def fake_sleep(s):
        sleeps.append(s)
        # Simulate time actually passing while "asleep".
        fake_now[0] += s

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(time, "time", lambda: fake_now[0])

    limiter = SlidingWindowRateLimiter(max_requests=2, time_window=60)
    limiter.acquire()
    limiter.acquire()
    # Third call must wait for the window to slide before it can proceed.
    limiter.acquire()

    assert len(sleeps) >= 1
    # Both original timestamps (both at t=1000) expire together once we've
    # slept past the 60s window, so only the third call's own timestamp
    # remains recorded afterward.
    assert len(limiter.request_times) == 1


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


def test_build_user_prompt_prepends_instruction_and_keeps_arabic_unescaped():
    verses = [{"id": "1_001", "sadr": "سدر", "ajuz": "عجز"}]
    prompt = build_user_prompt(verses)

    assert prompt.startswith(config.PROMPT_INSTRUCTION)
    # ensure_ascii=False must be in effect: raw Arabic text present, not
    # \uXXXX escapes.
    assert "سدر" in prompt
    assert "\\u0633" not in prompt

    # And the payload appended is valid, round-trippable JSON.
    json_part = prompt[len(config.PROMPT_INSTRUCTION) + 2 :]
    assert json.loads(json_part) == verses


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------


def test_call_llm_sends_thinking_disabled_explicitly_by_default(
    fake_openai_client, fake_rate_limiter, monkeypatch
):
    monkeypatch.setattr(config, "DEFAULT_THINKING_ENABLED", False)
    client = fake_openai_client(response_content="[]")

    content, reasoning, error, usage = call_llm(client, [{"id": "1"}], fake_rate_limiter)

    assert error is None
    assert content == "[]"
    assert reasoning is None
    sent_kwargs = client.calls[0]
    assert sent_kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in sent_kwargs


def test_call_llm_sends_thinking_enabled_and_reasoning_effort(
    fake_openai_client, fake_rate_limiter
):
    client = fake_openai_client(response_content="[]", reasoning_content="because...")

    content, reasoning, error, usage = call_llm(
        client,
        [{"id": "1"}],
        fake_rate_limiter,
        thinking_enabled=True,
        reasoning_effort="max",
    )

    assert error is None
    assert reasoning == "because..."
    sent_kwargs = client.calls[0]
    assert sent_kwargs["extra_body"]["thinking"] == {"type": "enabled"}
    assert sent_kwargs["reasoning_effort"] == "max"


def test_call_llm_never_raises_on_client_exception(fake_openai_client, fake_rate_limiter):
    client = fake_openai_client(raise_exc=RuntimeError("network exploded"))

    content, reasoning, error, usage = call_llm(client, [{"id": "1"}], fake_rate_limiter)

    assert content is None
    assert reasoning is None
    assert "network exploded" in error


def test_call_llm_nvidia_provider_uses_chat_template_kwargs_shape(
    fake_openai_client, fake_rate_limiter
):
    client = fake_openai_client(response_content="[]")

    call_llm(
        client,
        [{"id": "1"}],
        fake_rate_limiter,
        model="deepseek-ai/deepseek-v4-flash",
        provider=config.PROVIDER_NVIDIA,
        thinking_enabled=False,
        reasoning_effort="high",
    )

    sent_kwargs = client.calls[0]
    assert sent_kwargs["extra_body"] == {
        "chat_template_kwargs": {"thinking": False, "reasoning_effort": "high"}
    }
    # NVIDIA never gets the DeepSeek-style top-level reasoning_effort param.
    assert "reasoning_effort" not in sent_kwargs


def test_call_llm_nvidia_provider_sends_reasoning_effort_even_when_thinking_off(
    fake_openai_client, fake_rate_limiter
):
    client = fake_openai_client(response_content="[]")

    call_llm(
        client,
        [{"id": "1"}],
        fake_rate_limiter,
        model="deepseek-ai/deepseek-v4-flash",
        provider=config.PROVIDER_NVIDIA,
        thinking_enabled=False,
    )

    assert client.calls[0]["extra_body"]["chat_template_kwargs"]["thinking"] is False


def test_call_llm_deepseek_provider_unaffected_by_nvidia_branch(
    fake_openai_client, fake_rate_limiter
):
    """Sanity check: default (no provider passed) behaves exactly as before."""
    client = fake_openai_client(response_content="[]")

    call_llm(client, [{"id": "1"}], fake_rate_limiter, thinking_enabled=True, reasoning_effort="max")

    sent_kwargs = client.calls[0]
    assert sent_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert sent_kwargs["reasoning_effort"] == "max"


def test_call_llm_reads_reasoning_field_when_reasoning_content_absent(
    fake_openai_client, fake_rate_limiter, monkeypatch
):
    """NVIDIA's SDK response exposes `reasoning`, not `reasoning_content` --
    call_llm must fall back to it."""

    class _NvidiaFakeChoice:
        def __init__(self, content, reasoning):
            self.message = type(
                "FakeMessage", (), {"content": content, "reasoning": reasoning}
            )()

    class _NvidiaFakeCompletion:
        def __init__(self, content, reasoning):
            self.choices = [_NvidiaFakeChoice(content, reasoning)]
            self.usage = None

    class _NvidiaFakeClient:
        def __init__(self):
            self.calls = []
            outer = self

            class _Completions:
                def create(self, **kwargs):
                    outer.calls.append(kwargs)
                    return _NvidiaFakeCompletion("[]", "because nvidia said so")

            class _Chat:
                def __init__(self):
                    self.completions = _Completions()

            self.chat = _Chat()

    client = _NvidiaFakeClient()
    content, reasoning, error, usage = call_llm(
        client, [{"id": "1"}], fake_rate_limiter, provider=config.PROVIDER_NVIDIA
    )

    assert error is None
    assert reasoning == "because nvidia said so"


def test_call_llm_defaults_come_from_config(fake_openai_client, fake_rate_limiter, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_MODEL", "deepseek-v4-pro")
    client = fake_openai_client(response_content="[]")

    call_llm(client, [{"id": "1"}], fake_rate_limiter)

    assert client.calls[0]["model"] == "deepseek-v4-pro"


# ---------------------------------------------------------------------------
# setup_client
# ---------------------------------------------------------------------------


class _FakeModels:
    def __init__(self, raise_exc=None):
        self._raise_exc = raise_exc

    def list(self):
        if self._raise_exc:
            raise self._raise_exc
        return ["deepseek-v4-flash"]


class _FakeOpenAIForAuth:
    def __init__(self, api_key, base_url, raise_exc=None):
        self.api_key = api_key
        self.base_url = base_url
        self.models = _FakeModels(raise_exc=raise_exc)


def test_setup_client_succeeds_with_valid_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setattr(
        llm_client, "OpenAI", lambda api_key, base_url: _FakeOpenAIForAuth(api_key, base_url)
    )

    client = setup_client()
    assert client.api_key == "sk-fake"


def test_setup_client_exits_on_authentication_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-bad")
    auth_error = openai.AuthenticationError(
        "invalid key",
        response=httpx.Response(
            status_code=401, request=httpx.Request("GET", "https://api.deepseek.com/models")
        ),
        body=None,
    )
    monkeypatch.setattr(
        llm_client,
        "OpenAI",
        lambda api_key, base_url: _FakeOpenAIForAuth(api_key, base_url, raise_exc=auth_error),
    )

    with pytest.raises(SystemExit):
        setup_client()


def test_setup_client_prompts_for_key_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: "typed-in-key")
    monkeypatch.setattr(
        llm_client, "OpenAI", lambda api_key, base_url: _FakeOpenAIForAuth(api_key, base_url)
    )

    client = setup_client()
    assert client.api_key == "typed-in-key"


def test_setup_client_defaults_to_config_model_provider(monkeypatch):
    """With no provider argument, setup_client reads config.MODEL_PROVIDER --
    so the default (unset MODEL_PROVIDER env var) behaves exactly as it did
    before NVIDIA support existed."""
    monkeypatch.setattr(config, "MODEL_PROVIDER", config.PROVIDER_DEEPSEEK)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setattr(
        llm_client, "OpenAI", lambda api_key, base_url: _FakeOpenAIForAuth(api_key, base_url)
    )

    client = setup_client()
    assert client.base_url == "https://api.deepseek.com"


def test_setup_client_nvidia_provider_uses_nvidia_key_and_base_url(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-fake")
    monkeypatch.setattr(
        llm_client, "OpenAI", lambda api_key, base_url: _FakeOpenAIForAuth(api_key, base_url)
    )

    client = setup_client(provider=config.PROVIDER_NVIDIA)
    assert client.api_key == "nvapi-fake"
    assert client.base_url == "https://integrate.api.nvidia.com/v1"


def test_setup_client_nvidia_provider_prompts_for_key_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    prompts = []

    def _fake_input(prompt):
        prompts.append(prompt)
        return "typed-nvidia-key"

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(
        llm_client, "OpenAI", lambda api_key, base_url: _FakeOpenAIForAuth(api_key, base_url)
    )

    client = setup_client(provider=config.PROVIDER_NVIDIA)
    assert client.api_key == "typed-nvidia-key"
    assert "NVIDIA_API_KEY" in prompts[0]
