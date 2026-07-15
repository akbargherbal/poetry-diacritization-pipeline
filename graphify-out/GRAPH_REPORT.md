# Graph Report - .  (2026-07-15)

## Corpus Check
- Corpus is ~12,135 words - fits in a single context window. You may not need a graph.

## Summary
- 237 nodes · 511 edges · 16 communities (10 shown, 6 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 65 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- LLM Client & Rate Limiter
- CLI Commands & Registry Hub
- Validation Pass & Registry Tests
- Validation Logic & Core Tests
- CLI Integration Tests
- Generation Phase & Tests
- Text Normalization & Tests
- Test Fixtures & Mocks
- Diacritization Output Export
- Documentation & Dev Dependencies
- Runtime Directory Isolation
- PyArud Scoring Docs
- Fidelity Check Docs
- OpenAI Package Dependency
- Pandas Package Dependency
- PyArabic Package Dependency

## God Nodes (most connected - your core abstractions)
1. `make_registry_df()` - 45 edges
2. `run_validation_pass()` - 22 edges
3. `normalize()` - 20 edges
4. `run_generation_pass()` - 18 edges
5. `call_llm()` - 17 edges
6. `SlidingWindowRateLimiter` - 15 edges
7. `_run_cli()` - 14 edges
8. `_parse_llm_json()` - 13 edges
9. `load_or_build_registry()` - 12 edges
10. `save_registry()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `run_generation_pass()` --references--> `config.MODEL`  [EXTRACTED]
  poetry_diacritization/generate.py → docs/TESTING_STRATEGY.md
- `_process_one_batch` --calls--> `call_llm()`  [EXTRACTED]
  docs/TESTING_STRATEGY.md → poetry_diacritization/llm_client.py
- `_process_one_batch` --calls--> `save_registry()`  [EXTRACTED]
  docs/TESTING_STRATEGY.md → poetry_diacritization/registry.py
- `run_validation_pass()` --calls--> `_parse_llm_json`  [EXTRACTED]
  poetry_diacritization/validate.py → docs/TESTING_STRATEGY.md
- `run_validation_pass()` --calls--> `_score_with_pyarud`  [EXTRACTED]
  poetry_diacritization/validate.py → docs/TESTING_STRATEGY.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Pipeline Execution Flow (run.py / cli.py)** — poetry_diacritization_cli_cmd_generate, poetry_diacritization_validate_run_validation_pass, poetry_diacritization_threshold_apply_threshold, poetry_diacritization_export_export_passed [EXTRACTED 1.00]
- **Registry Operations Flow** — poetry_diacritization_registry_build_registry_from_input, poetry_diacritization_registry_make_batches, poetry_diacritization_registry_save_registry [INFERRED 0.95]

## Communities (16 total, 6 thin omitted)

### Community 0 - "LLM Client & Rate Limiter"
Cohesion: 0.10
Nodes (30): OpenAI, All tunable knobs live here. Nothing in the rest of the package should hardcode, # IMPORTANT: DeepSeek's API defaults thinking to ON. Disabling it requires, Stage 1 — Generate.  Calls the LLM for every batch of verses that still needs an, build_user_prompt(), call_llm(), Thread-safe rate limiter, sliding window, no lock-sleep bottleneck., One API call for one batch of verses (a single poem's worth, <= BATCH_SIZE). (+22 more)

### Community 1 - "CLI Commands & Registry Hub"
Cohesion: 0.13
Nodes (30): DataFrame, cmd_all(), cmd_export(), cmd_generate(), cmd_revalidate(), cmd_status(), cmd_threshold(), cmd_validate() (+22 more)

### Community 2 - "Validation Pass & Registry Tests"
Cohesion: 0.10
Nodes (32): _parse_llm_json, _score_with_pyarud, run_validation_pass(), fake_input_pickle(), make_registry_df(), Factory fixture: build a small registry DataFrame by hand.      Usage:         d, Factory fixture: write a minimal batch-level input pickle to disk.      Mirrors, # NOTE: build_registry_from_input's `input_pickle_path` default is bound (+24 more)

### Community 3 - "Validation Logic & Core Tests"
Cohesion: 0.12
Nodes (25): _parse_llm_json(), Stage 2 — Validate.  Runs entirely offline against whatever Stage 1 already save, Returns a dict {verse_id: {'sadr': ..., 'ajuz': ...}} or raises ValueError., Returns (score, detail_dict_or_None, error_or_None). Never raises., Send rows back through run_validation_pass without calling the LLM     again — f, reset_for_revalidation(), _score_with_pyarud(), _FakeProcessor (+17 more)

### Community 4 - "CLI Integration Tests"
Cohesion: 0.14
Nodes (16): Stub out every function main() can dispatch to, and record calls., _run_cli(), stub_pipeline_funcs(), test_all_command_runs_generate_validate_threshold(), test_export_command_dispatches(), test_generate_command_dispatches_with_defaults(), test_model_flag_passed_through(), test_no_save_reasoning_flag() (+8 more)

### Community 5 - "Generation Phase & Tests"
Cohesion: 0.19
Nodes (19): config.MODEL, _process_one_batch, _process_one_batch(), Run one generation pass over everything in df that needs an LLM attempt.      mo, run_generation_pass(), _save_raw_response(), _batch_for(), test_process_one_batch_failure_leaves_registry_untouched() (+11 more)

### Community 6 - "Text Normalization & Tests"
Cohesion: 0.24
Nodes (14): normalize(), One normalize() function. Applied identically to the original undiacritized text, texts_match(), test_alef_hamza_forms_normalize_to_bare_alef(), test_alef_maksura_does_not_collapse_into_plain_alef(), test_alef_maksura_normalized_when_flag_on(), test_alef_maksura_not_normalized_when_flag_off(), test_none_input_returns_empty_string() (+6 more)

### Community 7 - "Test Fixtures & Mocks"
Cohesion: 0.15
Nodes (10): fake_rate_limiter(), FakeChoice, FakeCompletion, FakeOpenAIClient, FakeRateLimiter, isolated_runtime_dir(), Shared fixtures for the poetry_diacritization test suite.  Nothing in this suite, Duck-typed stand-in for openai.OpenAI, for use in unit tests.      Records every (+2 more)

### Community 8 - "Diacritization Output Export"
Cohesion: 0.46
Nodes (6): export_passed(), test_export_passed_empty_set_still_writes_valid_files(), test_export_passed_only_includes_passed_rows(), test_export_passed_preserves_arabic_text_through_csv_round_trip(), test_export_passed_renames_current_columns(), test_export_passed_writes_pickle_and_csv()

### Community 9 - "Documentation & Dev Dependencies"
Cohesion: 0.29
Nodes (7): Testing Philosophy, LLM Role as Memory Recall, Poetry Diacritization Pipeline, Registry (registry.pkl), Three-Stage Pipeline Architecture, pytest, pytest-cov

## Knowledge Gaps
- **13 isolated node(s):** `Registry (registry.pkl)`, `Text Fidelity Check`, `pyarud Scoring`, `Isolated Runtime Directory Fixture`, `pandas` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_registry_df()` connect `Validation Pass & Registry Tests` to `CLI Commands & Registry Hub`, `Validation Logic & Core Tests`, `CLI Integration Tests`, `Generation Phase & Tests`, `Test Fixtures & Mocks`, `Diacritization Output Export`?**
  _High betweenness centrality (0.170) - this node is a cross-community bridge._
- **Why does `normalize()` connect `Text Normalization & Tests` to `CLI Commands & Registry Hub`, `Validation Pass & Registry Tests`, `Validation Logic & Core Tests`, `Test Fixtures & Mocks`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `run_validation_pass()` connect `Validation Pass & Registry Tests` to `CLI Commands & Registry Hub`, `Validation Logic & Core Tests`, `Text Normalization & Tests`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `make_registry_df()` (e.g. with `stub_pipeline_funcs()` and `test_export_passed_empty_set_still_writes_valid_files()`) actually correct?**
  _`make_registry_df()` has 43 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Registry (registry.pkl)`, `Text Fidelity Check`, `pyarud Scoring` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `LLM Client & Rate Limiter` be split into smaller, more focused modules?**
  _Cohesion score 0.09872241579558652 - nodes in this community are weakly interconnected._
- **Should `CLI Commands & Registry Hub` be split into smaller, more focused modules?**
  _Cohesion score 0.13445378151260504 - nodes in this community are weakly interconnected._