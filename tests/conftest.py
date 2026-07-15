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
        """
        rows = [
            {
                "poem_no": poem["poem_no"],
                "meter": poem["meter"],
                "DATA": poem["verses"],
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


class FakeCompletion:
    def __init__(self, content, reasoning_content=None):
        self.choices = [FakeChoice(content, reasoning_content)]


class FakeOpenAIClient:
    """Duck-typed stand-in for openai.OpenAI, for use in unit tests.

    Records every call's kwargs in `self.calls` so tests can assert on
    exactly what was sent (e.g. the `thinking` extra_body payload),
    without ever touching the network.
    """

    def __init__(self, response_content="{}", reasoning_content=None, raise_exc=None):
        self.calls = []
        self._response_content = response_content
        self._reasoning_content = reasoning_content
        self._raise_exc = raise_exc

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer._raise_exc is not None:
                    raise outer._raise_exc
                return FakeCompletion(outer._response_content, outer._reasoning_content)

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        self.chat = _Chat()


@pytest.fixture
def fake_openai_client():
    """Factory fixture: build a FakeOpenAIClient with a canned response."""

    def _make(response_content="{}", reasoning_content=None, raise_exc=None):
        return FakeOpenAIClient(
            response_content=response_content,
            reasoning_content=reasoning_content,
            raise_exc=raise_exc,
        )

    return _make


class FakeRateLimiter:
    """A rate limiter stand-in whose acquire() never blocks."""

    def acquire(self):
        return None


@pytest.fixture
def fake_rate_limiter():
    return FakeRateLimiter()
