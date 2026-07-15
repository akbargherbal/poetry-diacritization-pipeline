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
MODEL = "deepseek-v4-pro"
MAX_TOKENS = 4000          # 12 short verses of JSON does not need 64k tokens
TEMPERATURE = 0.4
TOP_P = 0.9
# This task is memory-recall, not reasoning — keep thinking mode OFF by
# default so the model doesn't burn tokens deliberating on prosody rules.
# Flip to True if you want to A/B test whether thinking mode helps recall.
THINKING_ENABLED = False
REASONING_EFFORT = "high"  # only used if THINKING_ENABLED is True

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
