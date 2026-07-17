# Graph Report - .  (2026-07-17)

## Corpus Check
- 12 files · ~16,952 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 361 nodes · 686 edges · 21 communities (12 shown, 9 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 83 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- OpenAI LLM Client
- Config & Registry Export
- Fidelity Validation & Scoring
- LLM Verse Generation
- CLI Test Suite
- Git Checkpointing Pipeline
- CLI Interface Commands
- Text Normalization
- DeepSeek Batch Pricing
- CLI Mocking & Test Helpers
- Pricing Verification Tests
- Quality Threshold Filtering
- Configuration File Tests
- Graphify Knowledge Graph
- Pytest Dependency
- Pytest Coverage
- OpenAI Library
- Pandas Library
- Pyarabic Dependency
- Pyarud Dependency
- Run Pipeline Entrypoint

## God Nodes (most connected - your core abstractions)
1. `make_registry_df()` - 54 edges
2. `call_llm()` - 22 edges
3. `_run_cli()` - 20 edges
4. `normalize()` - 18 edges
5. `run_generation_pass()` - 17 edges
6. `run_validation_pass()` - 15 edges
7. `_process_one_batch()` - 15 edges
8. `SlidingWindowRateLimiter` - 15 edges
9. `setup_client()` - 14 edges
10. `_parse_llm_json()` - 13 edges

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

## Communities (21 total, 9 thin omitted)

### Community 0 - "OpenAI LLM Client"
Cohesion: 0.08
Nodes (42): OpenAI, _build_extra_body(), build_user_prompt(), call_llm(), _extract_usage(), The thinking/reasoning-effort knobs are conceptually the same across     provide, One API call for one batch of verses (a single poem's worth, <= BATCH_SIZE)., Thread-safe rate limiter, sliding window, no lock-sleep bottleneck. (+34 more)

### Community 1 - "Config & Registry Export"
Cohesion: 0.08
Nodes (36): All tunable knobs live here. Nothing in the rest of the package should hardcode, # IMPORTANT: DeepSeek's API defaults thinking to ON. Disabling it requires, Decide which batch-level pickle to build the registry from.      This used to be, resolve_input_pickle(), export_passed(), fake_input_pickle(), make_registry_df(), Factory fixture: write a minimal batch-level input pickle to disk.      Mirrors (+28 more)

### Community 2 - "Fidelity Validation & Scoring"
Cohesion: 0.10
Nodes (37): _parse_llm_json(), Returns a dict {verse_id: {'sadr': ..., 'ajuz': ...}} or raises ValueError., Returns (score, detail_dict_or_None, error_or_None). Never raises., Send rows back through run_validation_pass without calling the LLM     again — f, reset_for_revalidation(), run_validation_pass(), _score_with_pyarud(), _FakeProcessor (+29 more)

### Community 3 - "LLM Verse Generation"
Cohesion: 0.12
Nodes (30): _cost_fields_for_batch(), _process_one_batch(), Stage 1 — Generate.  Calls the LLM for every batch of verses that still needs an, Run one generation pass over everything in df that needs an LLM attempt.      mo, Turn one call's token usage into a *per-verse* share of that call's     tokens/c, run_generation_pass(), _save_raw_response(), _batch_for() (+22 more)

### Community 4 - "CLI Test Suite"
Cohesion: 0.11
Nodes (22): Stub out every function main() can dispatch to, and record calls., _run_cli(), stub_pipeline_funcs(), test_all_command_passes_checkpoint_args(), test_all_command_runs_generate_validate_threshold(), test_checkpoint_every_defaults_to_config(), test_checkpoint_every_flag_overrides_default(), test_export_command_dispatches() (+14 more)

### Community 5 - "Git Checkpointing Pipeline"
Cohesion: 0.12
Nodes (23): CompletedProcess, CheckpointCounter, push_checkpoint(), Periodic "checkpoint" commits + pushes of runtime/ during a long generation run,, Stage runtime/ (registry.pkl + raw_responses/), commit with `message`,     and p, Tracks completed batches across a generation pass (which may run     batches con, Call once per successfully completed batch. Pushes a checkpoint         when the, _run_git() (+15 more)

### Community 6 - "CLI Interface Commands"
Cohesion: 0.17
Nodes (24): DataFrame, cmd_all(), cmd_export(), cmd_generate(), cmd_revalidate(), cmd_status(), cmd_threshold(), cmd_validate() (+16 more)

### Community 7 - "Text Normalization"
Cohesion: 0.17
Nodes (18): normalize(), NORMALIZE_ALEF_MAKSURA, NORMALIZE_TEH, One normalize() function. Applied identically to the original undiacritized text, texts_match(), Stage 2 — Validate.  Runs entirely offline against whatever Stage 1 already save, Text Fidelity Check, test_alef_hamza_forms_normalize_to_bare_alef() (+10 more)

### Community 8 - "DeepSeek Batch Pricing"
Cohesion: 0.12
Nodes (18): BATCH_SIZE, DEFAULT_MODEL, DEFAULT_THINKING_ENABLED, calculate_cost(), DeepSeek batch-cost pricing — the ONLY file you need to touch to keep cost track, load_pricing_config(), Batch-cost calculation for DeepSeek API calls.  This module is the only thing th, Load DEEPSEEK_PRICING from pricing_config.py, if it exists.      Returns {} — ne (+10 more)

### Community 9 - "CLI Mocking & Test Helpers"
Cohesion: 0.12
Nodes (12): fake_rate_limiter(), FakeChoice, FakeCompletion, FakeOpenAIClient, FakeRateLimiter, FakeUsage, isolated_runtime_dir(), Shared fixtures for the poetry_diacritization test suite.  Nothing in this suite (+4 more)

### Community 10 - "Pricing Verification Tests"
Cohesion: 0.13
Nodes (3): Tests for pricing.py — the DeepSeek cost-calculation layer.  Core guarantee unde, These module-level "warn once" sets must not leak state between tests., _reset_warned_state()

### Community 11 - "Quality Threshold Filtering"
Cohesion: 0.29
Nodes (10): apply_threshold(), Stage 3 — Threshold.  pyarud gives a continuous 0.0-1.0 score, not a pass/fail., Quick look at where your scores land, to help pick a cutoff., score_distribution(), test_apply_threshold_boundary_is_inclusive(), test_apply_threshold_ignores_rows_without_a_score(), test_apply_threshold_include_rescored_reconsiders_everything(), test_apply_threshold_only_touches_scored_rows_by_default() (+2 more)

## Knowledge Gaps
- **16 isolated node(s):** `pandas`, `openai`, `pyarabic`, `pyarud`, `pytest` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_registry_df()` connect `Config & Registry Export` to `Fidelity Validation & Scoring`, `LLM Verse Generation`, `CLI Test Suite`, `CLI Mocking & Test Helpers`, `Quality Threshold Filtering`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `normalize()` connect `Text Normalization` to `DeepSeek Batch Pricing`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `save_registry()` connect `CLI Interface Commands` to `Quality Threshold Filtering`, `Fidelity Validation & Scoring`, `LLM Verse Generation`, `Text Normalization`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 52 inferred relationships involving `make_registry_df()` (e.g. with `stub_pipeline_funcs()` and `test_input_flag_is_passed_to_load_or_build_registry()`) actually correct?**
  _`make_registry_df()` has 52 INFERRED edges - model-reasoned connections that need verification._
- **What connects `pandas`, `openai`, `pyarabic` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `OpenAI LLM Client` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Should `Config & Registry Export` be split into smaller, more focused modules?**
  _Cohesion score 0.07505285412262157 - nodes in this community are weakly interconnected._