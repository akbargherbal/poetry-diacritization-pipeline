"""
Shared fixtures for the poetry_diacritization test suite.

Nothing in this suite is allowed to touch the real project's `runtime/`
directory, and nothing is allowed to make a real network call (DeepSeek or
otherwise) unless it's explicitly marked `@pytest.mark.integration` and
DEEPSEEK_API_KEY is set. See docs/TESTING_STRATEGY.md for the reasoning
behind each of these fixtures.
"""
import pandas as pd
import pytest

from poetry_diacritization import config
from poetry_diacritization.registry import REGISTRY_COLUMNS


@pytest.fixture(autouse=True)
def isolated_runtime_dir(tmp_path, monkeypatch):
    """Redirect every config path into a per-test tmp_path.

    Autouse so every test gets a clean, disposable runtime directory
    without asking for it explicitly, and so a bug in one test can never
    corrupt the real project's runtime/registry.pkl or raw_responses/.
    """
    runtime_dir = tmp_path / "runtime"
    raw_responses_dir = runtime_dir / "raw_responses"
    raw_responses_dir.mkdir(parents=True)

    monkeypatch.setattr(config, "RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setattr(config, "REGISTRY_PATH", str(runtime_dir / "registry.pkl"))
    monkeypatch.setattr(config, "RAW_RESPONSES_DIR", str(raw_responses_dir))
    monkeypatch.setattr(config, "LOG_FILE", str(runtime_dir / "pipeline.log"))

    export_dir = tmp_path / "data"
    export_dir.mkdir()
    monkeypatch.setattr(config, "EXPORT_PATH", str(export_dir / "diacritized_verses.pkl"))
    monkeypatch.setattr(config, "EXPORT_CSV_PATH", str(export_dir / "diacritized_verses.csv"))

    yield runtime_dir


@pytest.fixture
def make_registry_df():
    """Factory fixture: build a small registry DataFrame by hand.

    Usage:
        df = make_registry_df([
            {"verse_id": "1_001", "poem_no": 1, "meter": "wafer",
             "sadr_raw": "...", "ajuz_raw": "...", "status": "pending"},
            ...
        ])

    Any REGISTRY_COLUMNS field not given for a row falls back to a sane
    default so tests only have to specify what they actually care about.
    """

    def _make(rows):
        defaults = {
            "verse_id": None,
            "poem_no": 1,
            "meter": "wafer",
            "sadr_raw": "سدر",
            "ajuz_raw": "عجز",
            "sadr_norm": None,
            "ajuz_norm": None,
            "sadr_current": None,
            "ajuz_current": None,
            "status": "pending",
            "pyarud_score": None,
            "pyarud_detail": None,
            "pass_count": 0,
            "last_model": None,
            "last_call_id": None,
            "input_tokens_cache_hit": None,
            "input_tokens_cache_miss": None,
            "output_tokens": None,
            "call_cost_usd": None,
            "cumulative_cost_usd": 0.0,
        }
        full_rows = []
        for row in rows:
            merged = {**defaults, **row}
            # sadr_norm/ajuz_norm default to normalize(raw) if not given,
            # so fidelity checks pass unless a test deliberately mismatches.
            from poetry_diacritization.text_normalize import normalize

            if merged["sadr_norm"] is None:
                merged["sadr_norm"] = normalize(merged["sadr_raw"])
            if merged["ajuz_norm"] is None:
                merged["ajuz_norm"] = normalize(merged["ajuz_raw"])
            full_rows.append(merged)

        df = pd.DataFrame(full_rows, columns=REGISTRY_COLUMNS)
        df = df.set_index("verse_id", drop=False)
        return df

    return _make


@pytest.fixture
def fake_input_pickle(tmp_path):
    """Factory fixture: write a minimal batch-level input pickle to disk.

    Mirrors the shape of data/SAMPLE_POEMS.pkl closely enough for
    build_registry_from_input, without depending on that real file.
    """

    def _make(poems, filename="fake_input.pkl"):
        """
        poems: list of dicts like
            {"poem_no": 1, "meter": "wafer", "verses": [
                {"verse_id": "1_001", "sadr": "...", "ajuz": "..."},
                ...
            ]}
        Any other keys on a poem dict besides poem_no/meter/verses (e.g.
        POET_NAME, POET_RANK) are carried through as extra batch-level
        columns on the input pickle, exactly like a real input pickle with
        columns beyond poem_no/meter/DATA.
        """
        rows = [
            {
                "poem_no": poem["poem_no"],
                "meter": poem["meter"],
                "DATA": poem["verses"],
                **{k: v for k, v in poem.items() if k not in ("poem_no", "meter", "verses")},
            }
            for poem in poems
        ]
        df = pd.DataFrame(rows)
        path = tmp_path / filename
        df.to_pickle(path, protocol=4)
        return str(path)

    return _make


class FakeChoice:
    def __init__(self, content, reasoning_content=None):
        self.message = type(
            "FakeMessage", (), {"content": content, "reasoning_content": reasoning_content}
        )()


class FakeUsage:
    """Duck-typed stand-in for the OpenAI-SDK usage object, DeepSeek-shaped
    (adds prompt_cache_hit_tokens / prompt_cache_miss_tokens on top of the
    standard prompt_tokens / completion_tokens)."""

    def __init__(
        self,
        prompt_tokens=0,
        completion_tokens=0,
        prompt_cache_hit_tokens=0,
        prompt_cache_miss_tokens=0,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_cache_hit_tokens = prompt_cache_hit_tokens
        self.prompt_cache_miss_tokens = prompt_cache_miss_tokens


class FakeCompletion:
    def __init__(self, content, reasoning_content=None, usage=None):
        self.choices = [FakeChoice(content, reasoning_content)]
        self.usage = usage


class FakeOpenAIClient:
    """Duck-typed stand-in for openai.OpenAI, for use in unit tests.

    Records every call's kwargs in `self.calls` so tests can assert on
    exactly what was sent (e.g. the `thinking` extra_body payload),
    without ever touching the network.
    """

    def __init__(self, response_content="{}", reasoning_content=None, raise_exc=None, usage=None):
        self.calls = []
        self._response_content = response_content
        self._reasoning_content = reasoning_content
        self._raise_exc = raise_exc
        self._usage = usage

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer._raise_exc is not None:
                    raise outer._raise_exc
                return FakeCompletion(outer._response_content, outer._reasoning_content, outer._usage)

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        self.chat = _Chat()


@pytest.fixture
def fake_openai_client():
    """Factory fixture: build a FakeOpenAIClient with a canned response.

    Pass `usage=FakeUsage(...)` (imported from this module) to simulate a
    DeepSeek response's token-usage block for cost-tracking tests.
    """

    def _make(response_content="{}", reasoning_content=None, raise_exc=None, usage=None):
        return FakeOpenAIClient(
            response_content=response_content,
            reasoning_content=reasoning_content,
            raise_exc=raise_exc,
            usage=usage,
        )

    return _make


class FakeRateLimiter:
    """A rate limiter stand-in whose acquire() never blocks."""

    def acquire(self):
        return None


@pytest.fixture
def fake_rate_limiter():
    return FakeRateLimiter()
