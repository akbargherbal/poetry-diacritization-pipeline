# Graph Report - .  (2026-07-15)

## Corpus Check
- 13 files · ~13,465 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 293 nodes · 591 edges · 16 communities (9 shown, 7 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 75 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Validation Pipeline
- Generation Pipeline
- CLI Interface
- LLM Integration Client
- Configuration & Exporting
- Text Normalization
- CLI Unit Tests
- Test Fixtures & Mocks
- Pricing Logic Tests
- Pytest Dependency
- Coverage Dependency
- OpenAI Dependency
- Pandas Dependency
- PyArabic Dependency
- PyArud Dependency
- Run Status Utility

## God Nodes (most connected - your core abstractions)
1. `make_registry_df()` - 53 edges
2. `normalize()` - 20 edges
3. `run_validation_pass()` - 19 edges
4. `call_llm()` - 17 edges
5. `SlidingWindowRateLimiter` - 15 edges
6. `_run_cli()` - 14 edges
7. `_process_one_batch()` - 14 edges
8. `run_generation_pass()` - 14 edges
9. `_parse_llm_json()` - 13 edges
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
- `test_build_registry_from_input_normalizes_text()` --calls--> `normalize()`  [EXTRACTED]
  tests/test_registry.py → poetry_diacritization/text_normalize.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Pipeline Execution Stages** — run_generate, run_validate, run_threshold [EXTRACTED 1.00]
- **Pipeline Configuration Constants** — poetry_diacritization_config_default_model, poetry_diacritization_config_default_thinking_enabled, poetry_diacritization_config_batch_size [EXTRACTED 1.00]

## Communities (16 total, 7 thin omitted)

### Community 0 - "Validation Pipeline"
Cohesion: 0.09
Nodes (38): _parse_llm_json(), Stage 2 — Validate.  Runs entirely offline against whatever Stage 1 already save, Returns a dict {verse_id: {'sadr': ..., 'ajuz': ...}} or raises ValueError., Returns (score, detail_dict_or_None, error_or_None). Never raises., Send rows back through run_validation_pass without calling the LLM     again — f, reset_for_revalidation(), run_validation_pass(), _score_with_pyarud() (+30 more)

### Community 1 - "Generation Pipeline"
Cohesion: 0.10
Nodes (36): _cost_fields_for_batch(), _process_one_batch(), Stage 1 — Generate.  Calls the LLM for every batch of verses that still needs an, Run one generation pass over everything in df that needs an LLM attempt.      mo, Turn one call's token usage into a *per-verse* share of that call's     tokens/c, run_generation_pass(), _save_raw_response(), calculate_cost() (+28 more)

### Community 2 - "CLI Interface"
Cohesion: 0.12
Nodes (34): DataFrame, cmd_all(), cmd_export(), cmd_generate(), cmd_revalidate(), cmd_status(), cmd_threshold(), cmd_validate() (+26 more)

### Community 3 - "LLM Integration Client"
Cohesion: 0.11
Nodes (29): OpenAI, build_user_prompt(), call_llm(), _extract_usage(), One API call for one batch of verses (a single poem's worth, <= BATCH_SIZE)., Thread-safe rate limiter, sliding window, no lock-sleep bottleneck., Pull token counts out of the API response's `usage` block, for cost     tracking, setup_client() (+21 more)

### Community 4 - "Configuration & Exporting"
Cohesion: 0.09
Nodes (30): All tunable knobs live here. Nothing in the rest of the package should hardcode, # IMPORTANT: DeepSeek's API defaults thinking to ON. Disabling it requires, export_passed(), fake_input_pickle(), make_registry_df(), Factory fixture: write a minimal batch-level input pickle to disk.      Mirrors, Factory fixture: build a small registry DataFrame by hand.      Usage:         d, test_export_passed_empty_set_still_writes_valid_files() (+22 more)

### Community 5 - "Text Normalization"
Cohesion: 0.10
Nodes (29): BATCH_SIZE, DEFAULT_MODEL, DEFAULT_THINKING_ENABLED, normalize(), NORMALIZE_ALEF_MAKSURA, NORMALIZE_TEH, One normalize() function. Applied identically to the original undiacritized text, texts_match() (+21 more)

### Community 6 - "CLI Unit Tests"
Cohesion: 0.14
Nodes (16): Stub out every function main() can dispatch to, and record calls., _run_cli(), stub_pipeline_funcs(), test_all_command_runs_generate_validate_threshold(), test_export_command_dispatches(), test_generate_command_dispatches_with_defaults(), test_model_flag_passed_through(), test_no_save_reasoning_flag() (+8 more)

### Community 7 - "Test Fixtures & Mocks"
Cohesion: 0.12
Nodes (12): fake_rate_limiter(), FakeChoice, FakeCompletion, FakeOpenAIClient, FakeRateLimiter, FakeUsage, isolated_runtime_dir(), Shared fixtures for the poetry_diacritization test suite.  Nothing in this suite (+4 more)

### Community 8 - "Pricing Logic Tests"
Cohesion: 0.13
Nodes (3): Tests for pricing.py — the DeepSeek cost-calculation layer.  Core guarantee unde, These module-level "warn once" sets must not leak state between tests., _reset_warned_state()

## Knowledge Gaps
- **14 isolated node(s):** `pandas`, `openai`, `pyarabic`, `pyarud`, `pytest` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_registry_df()` connect `Configuration & Exporting` to `Validation Pipeline`, `Generation Pipeline`, `CLI Interface`, `CLI Unit Tests`, `Test Fixtures & Mocks`?**
  _High betweenness centrality (0.168) - this node is a cross-community bridge._
- **Why does `normalize()` connect `Text Normalization` to `Validation Pipeline`, `Configuration & Exporting`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Are the 51 inferred relationships involving `make_registry_df()` (e.g. with `stub_pipeline_funcs()` and `test_export_passed_empty_set_still_writes_valid_files()`) actually correct?**
  _`make_registry_df()` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `SlidingWindowRateLimiter` (e.g. with `_FakeModels` and `_FakeOpenAIForAuth`) actually correct?**
  _`SlidingWindowRateLimiter` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `pandas`, `openai`, `pyarabic` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Validation Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.09413067552602436 - nodes in this community are weakly interconnected._
- **Should `Generation Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.0951219512195122 - nodes in this community are weakly interconnected._