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


def setup_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        api_key = input("DEEPSEEK_API_KEY not set. Paste your DeepSeek API key: ").strip()

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        client.models.list()
        log.info("DeepSeek API authentication successful.")
    except openai.AuthenticationError:
        log.error("Authentication failed — check your API key.")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error contacting DeepSeek API: {e}")
        sys.exit(1)

    return client


def build_user_prompt(verses: list) -> str:
    payload = json.dumps(verses, ensure_ascii=False)
    return f"{config.PROMPT_INSTRUCTION}\n\n{payload}"


def call_llm(client: OpenAI, verses: list, rate_limiter: SlidingWindowRateLimiter):
    """
    One API call for one batch of verses (a single poem's worth, <= BATCH_SIZE).
    No system prompt, by design. Returns (raw_text, reasoning_text_or_None, error_or_None).
    Never raises — network/API failure is reported back as an error string so
    the caller can mark those verse_ids for retry instead of crashing the pass.
    """
    rate_limiter.acquire()
    prompt_text = build_user_prompt(verses)

    kwargs = dict(
        model=config.MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
    )
    if config.THINKING_ENABLED:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        return content, reasoning, None
    except Exception as e:
        log.error(f"LLM call failed for poem batch: {e}")
        return None, None, str(e)
