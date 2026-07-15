# Testing Strategy

This document describes how to write and organize tests for
`poetry_diacritization`. It uses `pytest`. The full suite described below
now exists in `tests/`.

## Findings from implementing this suite

Writing these tests surfaced two real bugs in `generate.py`/`cli.py` that
would have crashed every real `generate`/`all` CLI run before an API key
was even checked. Both are fixed as part of adding this test suite:

1. **`config.MODEL` didn't exist.** `_process_one_batch` recorded
   `df.loc[verse_ids, "last_model"] = config.MODEL`, but `config.py` only
   ever defined `DEFAULT_MODEL`/`SUPPORTED_MODELS` — `config.MODEL` would
   raise `AttributeError` on the very first successful batch.
2. **`run_generation_pass` never accepted the arguments `cli.py` was
   already passing it.** `cmd_generate`/`cmd_all` called
   `run_generation_pass(df, model=..., thinking_enabled=..., reasoning_effort=...)`,
   but the function's signature was only `(df, client=None)` — this would
   raise `TypeError` before a single batch ran.

The fix threads `model`/`thinking_enabled`/`reasoning_effort` through
`run_generation_pass` → `_process_one_batch` → `call_llm`, resolving
defaults from `config.py` once per pass, and records the model actually
used in `last_model`. See `test_generate.py::test_run_generation_pass_records_last_model_used`
and the two tests around it for the regression coverage.

A third, smaller thing worth knowing about (not a bug, tested as-is): a
pyarud scoring failure in `validate.py` is reported with the same
`"failed_parse"` status as a genuine JSON-parsing failure. It's a bit
misleading when reading the registry later, but changing it is a product
decision, not a test-suite fix — flagged here rather than changed
silently.

## Philosophy

The pipeline is a five-stage state machine (`generate` → `validate` →
`threshold` → `export`, plus `status`) built around one shared artifact: the
`registry` DataFrame. Almost every bug that matters here is a **state
transition** bug — a verse ending up in the wrong `status`, or a stage
reading a file the previous stage didn't actually write. Tests should be
built around that reality:

- **Prefer fast, offline, deterministic unit tests.** Nothing in the test
  suite should make a real network call — not to DeepSeek, not to
  anywhere. `llm_client.call_llm` and `pyarud.ArudhProcessor` are always
  mocked.
- **Test each stage's contract, not its implementation.** `validate.py`
  doesn't care how `generate.py` produced `runtime/raw_responses/<id>.txt`,
  only that it exists and has a certain shape. Tests should reflect that
  same boundary — build a fake raw response file, don't call `generate`
  to produce one.
- **The registry is the shared contract.** Most tests will build a small
  in-memory (or `tmp_path`-backed) registry DataFrame by hand rather than
  loading the real `data/SAMPLE_POEMS.pkl`, so each test only has the rows
  it needs to make its point.
- **Arabic text edge cases matter more than volume.** `text_normalize.py`
  and the fidelity check in `validate.py` are where correctness actually
  lives; they deserve the most parametrized cases, even though they're the
  smallest files.

## Layout

```
poetry-diacritization-pipeline/
├── poetry_diacritization/
│   └── ...
├── tests/
│   ├── conftest.py            # shared fixtures (tmp registry, fake config paths, etc.)
│   ├── test_text_normalize.py
│   ├── test_registry.py
│   ├── test_llm_client.py
│   ├── test_generate.py
│   ├── test_validate.py
│   ├── test_threshold.py
│   ├── test_export.py
│   └── test_cli.py
├── pytest.ini
└── ...
```

One test file per source module. `tests/` mirrors `poetry_diacritization/`
1:1 so it's always obvious where a new test belongs.

## Setup

`requirements.txt` doesn't include test dependencies — keep those separate
so a production install doesn't pull in `pytest`.

```
# requirements-dev.txt
-r requirements.txt
pytest
pytest-cov
```

`pytest.ini` at the repo root:

```ini
[pytest]
testpaths = tests
markers =
    slow: tests that exercise many iterations or larger fixtures
addopts = -ra --strict-markers
```

Run the suite with:

```bash
pytest                      # everything
pytest -m "not slow"        # skip the slower tests during inner-loop dev
pytest --cov=poetry_diacritization --cov-report=term-missing
```

## Isolating the filesystem (`config.RUNTIME_DIR`)

`config.py` computes `RUNTIME_DIR`, `REGISTRY_PATH`, and
`RAW_RESPONSES_DIR` at import time from `BASE_DIR`, and creates the
directories as a side effect of import. Tests must never write into the
real project's `runtime/` folder. The standard fixture for this,
in `conftest.py`:

```python
import pytest
from poetry_diacritization import config

@pytest.fixture(autouse=True)
def isolated_runtime_dir(tmp_path, monkeypatch):
    """Redirect every config path into a per-test tmp_path."""
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(config, "RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setattr(config, "REGISTRY_PATH", str(runtime_dir / "registry.pkl"))
    monkeypatch.setattr(config, "RAW_RESPONSES_DIR", str(runtime_dir / "raw_responses"))
    monkeypatch.setattr(config, "LOG_FILE", str(runtime_dir / "pipeline.log"))
    runtime_dir.mkdir()
    (runtime_dir / "raw_responses").mkdir()
    yield runtime_dir
```

Because it's `autouse`, every test gets a clean, disposable runtime
directory without asking for it explicitly.

## Module-by-module plan

### `text_normalize.py`

Pure function, no I/O, no mocking — the easiest and highest-value tests
in the suite. Parametrize heavily:

- Diacritics (tashkeel) are stripped.
- Tatweel (kashida) is stripped.
- Ligature and alef-hamza forms normalize to the same base letter, in the
  order the module's docstring warns about (alef before hamza).
- `NORMALIZE_TEH` toggled on/off (monkeypatch the module constant) to
  confirm teh-marbuta/heh collapsing behaves as documented.
- Whitespace collapsing: multiple spaces, leading/trailing whitespace.
- `None` input returns `""` rather than raising.
- `texts_match()` is a thin wrapper — one or two tests confirming it
  delegates to `normalize()` are enough.

### `registry.py`

- `build_registry_from_input`: write a tiny fake pickle (2–3 poems, a
  handful of verses each, matching the real `DATA` list-of-dicts shape) to
  `tmp_path` and assert the exploded DataFrame has the right row count,
  columns, and initial `status="pending"` / `pass_count=0` defaults.
- Duplicate `verse_id` across poems must raise `ValueError` — this is a
  correctness-critical guard, give it its own test.
- `load_or_build_registry`: one test with no existing registry file (falls
  through to `build_registry_from_input` + saves it), one test with an
  existing registry file (loads it, doesn't rebuild — assert this by
  spying on `build_registry_from_input` and asserting it's *not* called).
- `save_registry`: confirms the atomic-write pattern — write to `.tmp`,
  `os.replace` over the real path — by checking no `.tmp` file survives
  and the content round-trips through `pd.read_pickle`.
- `make_batches`: the part worth the most test cases.
  - Only rows whose `status` is in `NEEDS_GENERATION_STATUSES` are
    included.
  - Verses from the same poem never split across batches unless the poem
    exceeds `batch_size`.
  - Verses from different poems never share a batch, even when a batch
    would otherwise have room.
  - Ordering within a poem follows the numeric suffix of `verse_id`, not
    DataFrame row order — build a registry with rows deliberately
    out of order and assert the batch's verse list comes back sorted.
  - Empty pending set returns `[]`.

### `llm_client.py`

Never call the real DeepSeek API. Mock the OpenAI client.

- `SlidingWindowRateLimiter`: monkeypatch `time.time` (or use
  `freezegun`/a small fake clock) to test that `acquire()` blocks once
  `max_requests` is hit within the window and releases after it slides.
  Keep `time.sleep` mocked too so this test runs instantly rather than
  actually sleeping.
- `build_user_prompt`: asserts the verse list round-trips through
  `json.dumps` with `ensure_ascii=False` (Arabic text must stay
  UTF-8, not become `\uXXXX` escapes) and that `PROMPT_INSTRUCTION` is
  prepended.
- `call_llm`: use a fake client — a small class or `unittest.mock.Mock`
  whose `.chat.completions.create(**kwargs)` returns a stubbed response
  object exposing `.choices[0].message.content` and optionally
  `.reasoning_content`. Cases to cover:
  - `thinking_enabled=False` (or omitted, falling back to
    `config.DEFAULT_THINKING_ENABLED`): assert `extra_body["thinking"]`
    is explicitly `{"type": "disabled"}` — this is called out as
    load-bearing in the source, don't let it regress silently — and that
    `reasoning_effort` is **not** sent as a kwarg.
  - `thinking_enabled=True`: assert `extra_body["thinking"]["type"] ==
    "enabled"` and `reasoning_effort` is passed through.
  - An exception raised by the mocked `.create()` is caught and returned
    as `(None, None, str(error))`, never re-raised.
  - `setup_client()` itself: mock `OpenAI` and `client.models.list()` to
    test the auth-success and `AuthenticationError` branches without a
    real key. Don't test the interactive `input()` prompt path beyond
    confirming it's only reached when the env var is unset.

### `generate.py`

- `_save_raw_response`: this is where the reasoning-artifact toggle
  (`config.SAVE_REASONING_ARTIFACTS`, default `False`) lives — give it
  explicit coverage:
  - `save_reasoning=False` with a non-empty `reasoning` string → only the
    `<call_id>.txt` file is written; no `_reasoning.txt` file appears.
  - `save_reasoning=True` with a non-empty `reasoning` string → both
    files are written, with the expected contents.
  - `reasoning=None` → no reasoning file regardless of the flag.
  - `content=None` → the main file is still written, empty.
- `_process_one_batch`: mock `call_llm` to return a canned
  `(content, reasoning, error)` tuple.
  - Success path: registry rows for the batch's `verse_ids` move to
    `status="awaiting_validation"`, `pass_count` increments,
    `last_call_id`/`last_model` are stamped, and `save_registry` is
    called (spy on it rather than hitting disk twice).
  - Error path (`error is not None`): registry rows are left completely
    untouched — assert equality with the pre-call DataFrame slice.
- `run_generation_pass`:
  - Empty `make_batches()` result → returns `df` unchanged, no client
    calls attempted.
  - `save_reasoning=None` (the default) resolves to
    `config.SAVE_REASONING_ARTIFACTS` — patch the config value in the
    test and confirm it propagates down to `_save_raw_response` (e.g. by
    checking which files exist afterward, not by mocking every layer).
  - An explicit `save_reasoning=True` passed to this function overrides
    the config default, even if the config default is `False`.
  - Mark any test that spins up the real `ThreadPoolExecutor` with
    several batches as `@pytest.mark.slow` if it starts feeling heavy;
    otherwise keep `MAX_WORKERS` small via monkeypatch.

### `validate.py`

This module has the most edge cases per line of any file in the package —
budget the most test time here.

- `_parse_llm_json`:
  - Plain JSON array → parses directly.
  - JSON wrapped in a ```` ```json ... ``` ```` fence → fence is stripped.
  - JSON object containing one list-valued key (some models wrap the
    array) → the list is extracted.
  - JSON object with no list-valued key → raises `ValueError`.
  - Not valid JSON at all → raises `ValueError` wrapping the
    `JSONDecodeError`.
  - Empty / whitespace-only string → raises `ValueError`.
  - Items missing `id`/`verse_id` are silently skipped, not fatal to the
    whole parse.
  - Both `id` and `verse_id` keys are accepted; `id` wins if both present
    (test the actual precedence in the source).
- `_score_with_pyarud`: mock `ArudhProcessor.process_poem` — never invoke
  the real prosody scorer in unit tests.
  - Returns `(score, detail, None)` on a clean result, with `detail`
    containing only `input_pattern`/`best_ref_pattern` (confirm the full
    foot-by-foot breakdown is *not* being kept).
  - A result dict containing an `"error"` key → returns
    `(None, None, error_message)`.
  - The mocked processor raising an exception → caught, returns
    `(None, None, str(exception))`, never propagates.
- `run_validation_pass`, using the `isolated_runtime_dir` fixture to write
  fake raw-response files directly into `RAW_RESPONSES_DIR`:
  - Nothing in `"awaiting_validation"` → no-op, returns `df` unchanged.
  - Missing raw-response file for a `call_id` → affected rows move to
    `"failed_parse"`.
  - Raw response with a verse missing from the parsed map → that
    verse's row moves to `"failed_parse"`.
  - Raw response where `normalize(candidate) != normalize(original)` →
    `"failed_text_mismatch"`.
  - Fidelity passes, `_score_with_pyarud` (mocked) succeeds →
    `"scored"`, with `pyarud_score`/`pyarud_detail` populated.
  - Fidelity passes, mocked pyarud call errors → `"failed_parse"`, with
    `pyarud_detail={"error": ...}` (note: this is a real
    behavior-worth-flagging case, since the status name doesn't
    obviously communicate "this failed at scoring, not parsing" — test
    it as specified, but see it as a candidate for a future source-code
    fix, not a test-suite workaround).
  - Multiple verses sharing one `call_id` are only parsed once (assert
    the JSON-parsing mock/spy is called once per unique `call_id`, not
    once per verse).

### `threshold.py`

- Default call (`include_rescored=False`) only touches rows currently
  `"scored"`; rows already `"passed"`/`"failed_prosody"` from an earlier
  run are left alone even if their score would flip under a new cutoff.
- `include_rescored=True` reconsiders every row with a non-null
  `pyarud_score`, regardless of current status — including flipping a
  previously `"passed"` row back to `"failed_prosody"` under a stricter
  cutoff.
- Boundary condition: a score exactly equal to `cutoff` passes (`>=`, not
  `>`).
- `score_distribution` returns `None` when nothing has a score yet, and a
  proper `pandas.Series.describe()` output otherwise.

### `export.py`

- Only `status == "passed"` rows are exported.
- Output columns are exactly `EXPORT_COLUMNS` with `sadr_current` /
  `ajuz_current` renamed to `sadr` / `ajuz`.
- Both the pickle and the CSV are written (check via `tmp_path` +
  `isolated_runtime_dir`-style monkeypatching of `config.EXPORT_PATH` /
  `config.EXPORT_CSV_PATH`), and the CSV round-trips the Arabic text
  correctly under `utf-8-sig`.
- Empty passed-set still writes valid (empty) files rather than raising.

### `cli.py`

Argument parsing is worth testing directly, without a real API key or
real pyarud installed — mock at the `poetry_diacritization.cli` import
boundary (patch `run_generation_pass`, `run_validation_pass`,
`apply_threshold`, `export_passed`, `load_or_build_registry`) and drive
`main()` via `sys.argv` + `monkeypatch`.

- Each subcommand (`status`, `generate`, `validate`, `threshold`, `all`,
  `export`) parses and dispatches to the right `cmd_*` function.
- `--thinking` / `--no-thinking` are mutually exclusive; both flags
  present should raise a `SystemExit` from argparse.
- `--save-reasoning` / `--no-save-reasoning` likewise mutually exclusive,
  and default to `config.SAVE_REASONING_ARTIFACTS` when neither is given
  — assert this by inspecting the parsed `Namespace`, not by running a
  real generation pass.
- `--reasoning-effort` only accepts `config.SUPPORTED_REASONING_EFFORTS`;
  an invalid choice raises `SystemExit` with a non-zero code.
- `threshold`/`all` commands pass `--cutoff` and `--include-rescored`
  through unchanged to `apply_threshold`.

## What's intentionally out of scope

- **Real DeepSeek API calls.** No test should require `DEEPSEEK_API_KEY`
  to pass. If you want an end-to-end smoke test against the live API,
  keep it as a separate, manually-run script outside `pytest`, not part
  of the suite CI runs.
- **Real pyarud scoring accuracy.** Whether a given verse actually
  scores 0.94 vs 0.87 on a real meter is a linguistics question, not a
  software one — that belongs to pyarud's own test suite. This project's
  tests only need to confirm the score gets plumbed through correctly.
- **`data/SAMPLE_POEMS.pkl` as a fixture.** It's a real, unversioned data
  file, not a fixture — never write a test around its exact contents. If
  a specific real-poem edge case needs coverage, copy just enough of it
  into a small hand-built fixture in `conftest.py`.

## Adding a new test

1. Find the module the behavior lives in; open the matching
   `tests/test_<module>.py` (create it if it doesn't exist yet, mirroring
   the layout above).
2. Reach for `isolated_runtime_dir` if the code under test touches
   `config.RUNTIME_DIR`/`REGISTRY_PATH`/`RAW_RESPONSES_DIR` — never let a
   test touch the real `runtime/` folder.
3. Mock `openai`'s client and `pyarud`'s `ArudhProcessor` at the call
   site; don't let either escape into a real network/library call.
4. Prefer building a minimal registry DataFrame by hand over loading
   `SAMPLE_POEMS.pkl`.
5. Run `pytest path/to/test_file.py -v` before running the full suite.
