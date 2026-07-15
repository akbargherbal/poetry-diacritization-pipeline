"""
All tunable knobs live here. Nothing in the rest of the package should
hardcode a path, model name, or threshold — change it here instead.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE = os.path.join(BASE_DIR, "data", "SAMPLE_POEMS.pkl")

RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
REGISTRY_PATH = os.path.join(RUNTIME_DIR, "registry.pkl")
RAW_RESPONSES_DIR = os.path.join(RUNTIME_DIR, "raw_responses")
LOG_FILE = os.path.join(RUNTIME_DIR, "pipeline.log")

# Where the final, cleaned, passed-only dataset gets written by `export`
EXPORT_PATH = os.path.join(BASE_DIR, "data", "diacritized_verses.pkl")
EXPORT_CSV_PATH = os.path.join(BASE_DIR, "data", "diacritized_verses.csv")

os.makedirs(RUNTIME_DIR, exist_ok=True)
os.makedirs(RAW_RESPONSES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
BATCH_SIZE = 12  # verses per LLM call, per poem. Your "sweet spot" number.

# ---------------------------------------------------------------------------
# LLM / DeepSeek settings
# ---------------------------------------------------------------------------
# DeepSeek's first-party API lineup (api-docs.deepseek.com), as of this
# writing. deepseek-v4-flash is the cheaper/faster default; deepseek-v4-pro
# is the stronger, more expensive one. Override per-run with `--model`.
SUPPORTED_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]
DEFAULT_MODEL = "deepseek-v4-flash"

MAX_TOKENS = 4000          # 12 short verses of JSON does not need 384k tokens
TEMPERATURE = 0.4          # ignored by DeepSeek when thinking mode is on
TOP_P = 0.9                # ignored by DeepSeek when thinking mode is on

# This task is memory-recall, not reasoning — keep thinking mode OFF by
# default so the model doesn't burn tokens deliberating on prosody rules.
# IMPORTANT: DeepSeek's API defaults thinking to ON. Disabling it requires
# explicitly sending {"thinking": {"type": "disabled"}} — the client always
# sends this explicitly (see llm_client.py), it's never left implicit.
# Override per-run with `--thinking` / `--no-thinking`.
DEFAULT_THINKING_ENABLED = False

# Only meaningful when thinking is enabled. DeepSeek's own compatibility
# mapping collapses "low"/"medium" -> "high" and "xhigh" -> "max", so those
# are the only two settings that actually behave differently — that's why
# only these two are exposed here rather than a four-tier low/med/high/max
# preset that would be partly cosmetic.
SUPPORTED_REASONING_EFFORTS = ["high", "max"]
DEFAULT_REASONING_EFFORT = "high"
# Override per-run with `--reasoning-effort`.

# Raw model output (the JSON verse response) is always saved to
# RAW_RESPONSES_DIR — validate.py depends on it being there.
# The *reasoning/thinking* trace (only ever present when thinking mode is
# on) is a separate, optional artifact. It's disabled by default: it's not
# needed for validation, and it can balloon RAW_RESPONSES_DIR quickly at any
# real batch volume. Override per-run with `--save-reasoning`.
SAVE_REASONING_ARTIFACTS = False

MAX_WORKERS = 6            # concurrent threads
REQUESTS_PER_MINUTE = 6    # rate limit

# Statuses that still need an LLM attempt (fed into the next generate pass)
NEEDS_GENERATION_STATUSES = (
    "pending",
    "failed_text_mismatch",
    "failed_parse",
    "failed_prosody",
)

# ---------------------------------------------------------------------------
# The prompt. Deliberately minimal — no persona, no system prompt, no meter
# feet spelled out. The bet is LLM memory/pattern recall, not reasoning.
# ---------------------------------------------------------------------------
PROMPT_INSTRUCTION = (
    "هذه الأبيات مقتطعة من قصيدة تم إزالة الحركات والتشكيل من على أحرفها، "
    "قم بتشكيلها تشكيلًا تامًا لتكون صحيحة عروضيًا. "
    "إجابتك يجب أن تكون على هيئة JSON، لا تكتب أي تعليقات أو ملاحظات قبل أو بعد الإجابة.\n\n"
    "أعد نفس عدد الأبيات وبنفس المعرفات (id) وبنفس بنية الحقول (id, sadr, ajuz)، "
    "لكن مع تشكيل الشطرين (sadr, ajuz) تشكيلًا تامًا."
)
