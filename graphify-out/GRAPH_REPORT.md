# Graph Report - .  (2026-07-17)

## Corpus Check
- 8 files · ~14,471 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 313 nodes · 578 edges · 19 communities (11 shown, 8 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Validation Logic & Metrics
- Text Generation & Cost Pricing
- LLM Client & Rate Limiting
- Configuration & Normalization
- Export & Thresholding
- Command Line Interface (CLI)
- CLI Tests
- Test Mocks & Conftest
- Config & Config Tests
- Registry & Registry Tests
- Pricing Tests
- Graphify Knowledge Graph & Rules
- Pytest Dependencies
- Pytest Coverage Dependencies
- OpenAI Client Dependencies
- Pandas/Dataframe Dependencies
- PyArabic Dependencies
- PyArud Dependencies
- Pipeline Execution Runner

## God Nodes (most connected - your core abstractions)
1. `make_registry_df()` - 43 edges
2. `normalize()` - 18 edges
3. `call_llm()` - 17 edges
4. `_run_cli()` - 16 edges
5. `run_validation_pass()` - 15 edges
6. `SlidingWindowRateLimiter` - 15 edges
7. `_process_one_batch()` - 14 edges
8. `_parse_llm_json()` - 13 edges
9. `run_generation_pass()` - 13 edges
10. `load_or_build_registry()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Text Fidelity Check` --conceptually_related_to--> `normalize()`  [INFERRED]
  README.md → poetry_diacritization/text_normalize.py
- `normalize()` --references--> `NORMALIZE_ALEF_MAKSURA`  [EXTRACTED]
  poetry_diacritization/text_normalize.py → README.md
- `normalize()` --references--> `NORMALIZE_TEH`  [EXTRACTED]
  poetry_diacritization/text_normalize.py → README.md
- `run.py validate` --references--> `normalize()`  [EXTRACTED]
  README.md → poetry_diacritization/text_normalize.py
- `test_alef_hamza_forms_normalize_to_bare_alef()` --calls--> `normalize()`  [EXTRACTED]
  tests/test_text_normalize.py → poetry_diacritization/text_normalize.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Pipeline Execution Stages** — run_generate, run_validate, run_threshold [EXTRACTED 1.00]
- **Pipeline Configuration Constants** — poetry_diacritization_config_default_model, poetry_diacritization_config_default_thinking_enabled, poetry_diacritization_config_batch_size [EXTRACTED 1.00]

## Communities (19 total, 8 thin omitted)

### Community 0 - "Validation Logic & Metrics"
Cohesion: 0.09
Nodes (38): _parse_llm_json(), Stage 2 — Validate.  Runs entirely offline against whatever Stage 1 already save, Returns a dict {verse_id: {'sadr': ..., 'ajuz': ...}} or raises ValueError., Returns (score, detail_dict_or_None, error_or_None). Never raises., Send rows back through run_validation_pass without calling the LLM     again — f, reset_for_revalidation(), run_validation_pass(), _score_with_pyarud() (+30 more)

### Community 1 - "Text Generation & Cost Pricing"
Cohesion: 0.10
Nodes (35): _cost_fields_for_batch(), _process_one_batch(), Stage 1 — Generate.  Calls the LLM for every batch of verses that still needs an, Run one generation pass over everything in df that needs an LLM attempt.      mo, Turn one call's token usage into a *per-verse* share of that call's     tokens/c, run_generation_pass(), _save_raw_response(), calculate_cost() (+27 more)

### Community 2 - "LLM Client & Rate Limiting"
Cohesion: 0.11
Nodes (29): OpenAI, build_user_prompt(), call_llm(), _extract_usage(), One API call for one batch of verses (a single poem's worth, <= BATCH_SIZE)., Thread-safe rate limiter, sliding window, no lock-sleep bottleneck., Pull token counts out of the API response's `usage` block, for cost     tracking, setup_client() (+21 more)

### Community 3 - "Configuration & Normalization"
Cohesion: 0.09
Nodes (30): BATCH_SIZE, DEFAULT_MODEL, DEFAULT_THINKING_ENABLED, DeepSeek batch-cost pricing — the ONLY file you need to touch to keep cost track, normalize(), NORMALIZE_ALEF_MAKSURA, NORMALIZE_TEH, One normalize() function. Applied identically to the original undiacritized text (+22 more)

### Community 4 - "Export & Thresholding"
Cohesion: 0.12
Nodes (23): export_passed(), apply_threshold(), Stage 3 — Threshold.  pyarud gives a continuous 0.0-1.0 score, not a pass/fail., Quick look at where your scores land, to help pick a cutoff., score_distribution(), make_registry_df(), Factory fixture: build a small registry DataFrame by hand.      Usage:         d, test_export_passed_empty_set_still_writes_valid_files() (+15 more)

### Community 5 - "Command Line Interface (CLI)"
Cohesion: 0.17
Nodes (24): DataFrame, cmd_all(), cmd_export(), cmd_generate(), cmd_revalidate(), cmd_status(), cmd_threshold(), cmd_validate() (+16 more)

### Community 6 - "CLI Tests"
Cohesion: 0.13
Nodes (18): Stub out every function main() can dispatch to, and record calls., _run_cli(), stub_pipeline_funcs(), test_all_command_runs_generate_validate_threshold(), test_export_command_dispatches(), test_generate_command_dispatches_with_defaults(), test_input_flag_is_passed_to_load_or_build_registry(), test_model_flag_passed_through() (+10 more)

### Community 7 - "Test Mocks & Conftest"
Cohesion: 0.11
Nodes (14): fake_input_pickle(), fake_rate_limiter(), FakeChoice, FakeCompletion, FakeOpenAIClient, FakeRateLimiter, FakeUsage, isolated_runtime_dir() (+6 more)

### Community 8 - "Config & Config Tests"
Cohesion: 0.12
Nodes (6): All tunable knobs live here. Nothing in the rest of the package should hardcode, # IMPORTANT: DeepSeek's API defaults thinking to ON. Disabling it requires, Decide which batch-level pickle to build the registry from.      This used to be, resolve_input_pickle(), isolated_data_dir(), Point config.BASE_DIR at a throwaway dir with its own data/ subfolder,     so we

### Community 9 - "Registry & Registry Tests"
Cohesion: 0.12
Nodes (3): # NOTE: build_registry_from_input's `input_pickle_path` default is bound, Regression test: swapping data/SAMPLE_POEMS.pkl for a differently     named pick, test_build_registry_from_input_auto_detects_renamed_pickle()

### Community 10 - "Pricing Tests"
Cohesion: 0.13
Nodes (3): Tests for pricing.py — the DeepSeek cost-calculation layer.  Core guarantee unde, These module-level "warn once" sets must not leak state between tests., _reset_warned_state()

## Knowledge Gaps
- **16 isolated node(s):** `pandas`, `openai`, `pyarabic`, `pyarud`, `pytest` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `normalize()` connect `Configuration & Normalization` to `Validation Logic & Metrics`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Why does `make_registry_df()` connect `Export & Thresholding` to `Validation Logic & Metrics`, `Text Generation & Cost Pricing`, `Test Mocks & Conftest`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `save_registry()` connect `Command Line Interface (CLI)` to `Validation Logic & Metrics`, `Text Generation & Cost Pricing`, `Export & Thresholding`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 41 inferred relationships involving `make_registry_df()` (e.g. with `test_export_passed_empty_set_still_writes_valid_files()` and `test_export_passed_only_includes_passed_rows()`) actually correct?**
  _`make_registry_df()` has 41 INFERRED edges - model-reasoned connections that need verification._
- **What connects `pandas`, `openai`, `pyarabic` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Validation Logic & Metrics` be split into smaller, more focused modules?**
  _Cohesion score 0.09413067552602436 - nodes in this community are weakly interconnected._
- **Should `Text Generation & Cost Pricing` be split into smaller, more focused modules?**
  _Cohesion score 0.10256410256410256 - nodes in this community are weakly interconnected._