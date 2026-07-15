# Poetry Diacritization Pipeline

A brute-force pipeline for diacritizing classical Arabic poem verses with an
LLM, verified with `pyarud`. Built around one idea: **the LLM's job is
memory recall, not prosody reasoning.** No system prompt, no meter feet
spelled out, no persona. A single, minimal instruction, run over as many
passes as you want, with a registry that tracks per-verse status so failed
verses (and only failed verses) get re-sent — potentially to a different
model — on the next pass.

## Quick start

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...

python run.py status      # see what's in the registry (builds it on first run)
python run.py generate     # Stage 1: call the LLM for everything pending
python run.py validate     # Stage 2: parse + fidelity-check + score, offline
python run.py threshold --cutoff 0.90   # Stage 3: turn scores into pass/fail
python run.py status      # see how it went

python run.py generate     # run it again — only picks up what's still not passed
python run.py validate
python run.py threshold --cutoff 0.90

python run.py export       # write the clean, passed-only dataset to data/
```

Or all at once: `python run.py all --cutoff 0.90` (runs generate → validate →
threshold in sequence). You'll still want to run `generate`/`validate`/
`threshold` separately across multiple passes, since that's the whole point
of the registry — but `all` is convenient for a first pass.

To try a different model on the next pass, edit `MODEL` in
`poetry_diacritization/config.py` and run `generate` again — it will
re-attempt exactly the verses that are still not `passed`.

## Why three separate stages, not one script

- **Generate** is the only stage that touches the network. It doesn't parse
  or judge the response at all — it just saves the raw text to disk and
  marks the registry rows `awaiting_validation`. This means a bug in the
  parser, or a decision to loosen a threshold, never costs you an API call
  to fix — you already have the raw response sitting in `runtime/raw_responses/`.
- **Validate** is pure offline computation (JSON parsing, text comparison,
  `pyarud` scoring). Re-run it as many times as you want.
- **Threshold** is the only place a pass/fail cutoff is applied to a
  `pyarud` score, and it's applied to the *stored* score. Changing your mind
  from 0.90 to 0.95 costs one function call, not a re-scrape.

## The registry (`runtime/registry.pkl`)

One row per verse, keyed by `verse_id`. This is the single source of truth
and the only thing carried forward between passes — the original input
(`data/SAMPLE_POEMS.pkl`) is never modified. It's exploded into a flat,
per-verse table once, on first run.

| column | meaning |
|---|---|
| `verse_id` | permanent primary key, e.g. `1980_024` |
| `poem_no`, `meter` | carried from the source batch |
| `sadr_raw`, `ajuz_raw` | original undiacritized hemistichs (never touched) |
| `sadr_norm`, `ajuz_norm` | pre-computed normalized form of the above, for fast fidelity checks |
| `sadr_current`, `ajuz_current` | latest diacritized candidate from the LLM |
| `status` | see below |
| `pyarud_score` | raw float from `pyarud`, no threshold baked in |
| `pyarud_detail` | small dict (input/reference bit-patterns), for debugging |
| `pass_count` | how many times an LLM has attempted this verse |
| `last_model`, `last_call_id` | which model, and which raw response file, produced the current candidate |

### Status lifecycle

```
pending ──generate──► awaiting_validation ──validate──► scored ──threshold──► passed
   ▲                                              │                    │
   │                                              ▼                    ▼
   └──────────────── failed_parse / failed_text_mismatch / failed_prosody
                (all four of these get re-sent on the next `generate`)
```

`failed_parse` covers: the whole response wasn't valid JSON, or this
specific `verse_id` was missing from an otherwise-valid response, or
`pyarud` itself threw an exception on this verse (a genuine edge case in
the library, not a verdict on the diacritization — treated the same as any
other "couldn't get a clean answer" case, so it's retried, not discarded).

## Batching

Verses that still need an attempt are grouped **per poem**, in their
original order, into batches of at most `BATCH_SIZE` (12 by default,
`poetry_diacritization/config.py`). A poem with only 3 straggler verses
left gets a batch of 3 — batches are never padded with verses from other
poems, so the LLM always sees a contiguous run from a single poem.

## The prompt

Deliberately minimal, no system prompt:

```
هذه الأبيات مقتطعة من قصيدة تم إزالة الحركات والتشكيل من على أحرفها، قم
بتشكيلها تشكيلًا تامًا لتكون صحيحة عروضيًا. إجابتك يجب أن تكون على هيئة
JSON، لا تكتب أي تعليقات أو ملاحظات قبل أو بعد الإجابة.

أعد نفس عدد الأبيات وبنفس المعرفات (id) وبنفس بنية الحقول (id, sadr, ajuz)،
لكن مع تشكيل الشطرين (sadr, ajuz) تشكيلًا تامًا.

[{"id": "1980_024", "sadr": "...", "ajuz": "..."}, ...]
```

The second paragraph (specifying the same `id`/`sadr`/`ajuz` structure back)
is the one addition beyond your original wording — it's schema
specification, not prosody reasoning, and it's what makes reliable JSON
parsing possible at all. Thinking mode is off by default
(`THINKING_ENABLED = False`) since the premise is recall, not reasoning —
flip it in `config.py` if you want to A/B test whether it helps.

## Text fidelity check — what's okay, what's not

The LLM must only add diacritics, never change the underlying letters. This
is checked with a single `normalize()` function
(`poetry_diacritization/text_normalize.py`) applied identically to the
original and the LLM's (de-diacritized) output; the two normalized strings
must be **exactly** equal — not fuzzy-matched, not edit-distance-tolerant.
That's deliberate: fuzzy matching is exactly the loophole that would let a
"creative fix" like `يصرخ` → `يستصرخ` slip through undetected.

Normalized away as harmless orthographic variance:
- Diacritics and tatweel (ـ) — the whole point, and pure cosmetics
- Alef forms: `أ إ آ ا` → `ا`
- Hamza-on-waw/yeh: `ؤ ئ ء` → `ء`
- `ة` / `ه` (toggle: `NORMALIZE_TEH` in `text_normalize.py`)
- Lam-alef ligatures, extra whitespace

Anything else — inserted/deleted/substituted letters, added prefixes,
merged or split words, reordered words, a missing or extra verse — fails
as `failed_text_mismatch`, and `pyarud` is never even called on it (no
point scoring a verse whose words changed).

## `pyarud` scoring

Each verse is scored individually against its poem's **labeled** meter
(trusted as ground truth from the source corpus — not re-detected across
all 16 meters). `pyarud` is known to be imperfect on edge cases; a low
score is treated as "not verified," not as proof the diacritization is
wrong, and a `pyarud` exception is treated the same way (`failed_parse`,
retried later) rather than crashing the run or being silently discarded.

No pass/fail threshold is hardcoded anywhere in `generate.py` or
`validate.py` — only `threshold.py` decides, and only when you ask it to.
Run `python run.py status` to see the score distribution before picking a
cutoff.

## Raw response files (`runtime/raw_responses/`)

Every LLM response, successful or not, is saved to disk as
`{call_id}.txt` (plus `{call_id}_reasoning.txt` if the model returned a
reasoning trace). Nothing is ever deleted automatically — storage is cheap,
a lost LLM response is not free to regenerate. `last_call_id` in the
registry points to the exact file behind any given verse's current
candidate, for debugging.

## Extending / tuning

- **Different model per pass**: change `MODEL` in `config.py`, run
  `generate` again.
- **Batch size**: `BATCH_SIZE` in `config.py`.
- **Rate limits / concurrency**: `REQUESTS_PER_MINUTE`, `MAX_WORKERS`.
- **Loosen/tighten the prosody bar**: `python run.py threshold --cutoff X`
  — safe to re-run with a new cutoff anytime; add `--include-rescored` to
  also re-evaluate verses that were previously marked `failed_prosody`
  under an older cutoff.
- **Re-check fidelity rules**: edit `NORMALIZE_TEH` or the `normalize()`
  pipeline in `text_normalize.py`, then just re-run `validate` — no API
  calls needed, since `sadr_current`/`ajuz_current` are already saved.

## Known limitations / things to watch

- `pyarud` scoring uses the corpus's labeled meter as ground truth. If the
  source corpus mislabels a poem's meter, every verse in it will score low
  regardless of how good the diacritization is — worth spot-checking if a
  whole poem is stuck at `failed_prosody`.
- The fidelity check is strict by design (see above) — expect some
  legitimately-fine diacritizations to get flagged `failed_text_mismatch`
  if the LLM makes an orthographic choice outside the normalized set (e.g.
  choosing to write a hamzat-al-wasl differently in a way not covered
  here). Inspect that bucket periodically; it's a candidate for loosening
  `normalize()`, not necessarily an LLM error.
- No automatic cap on retry passes — a verse can stay in
  `failed_text_mismatch`/`failed_prosody`/`failed_parse` forever if you
  keep re-running `generate`. That's intentional (your call on when to stop
  spending tokens on stragglers), but nothing here will warn you if a
  handful of verses are eating passes without progress — worth eyeballing
  `pass_count` for verses that are still failing after 3-4 attempts.
