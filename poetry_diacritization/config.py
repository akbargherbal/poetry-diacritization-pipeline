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

# Env var override: point at any input pickle without touching the CLI.
# Lowest priority after an explicit --input/path argument, higher priority
# than auto-detection. See resolve_input_pickle() below.
INPUT_PICKLE_ENV_VAR = "POETRY_DIACRITIZATION_INPUT"


def resolve_input_pickle(explicit=None):
    """Decide which batch-level pickle to build the registry from.

    This used to be a single hardcoded constant (INPUT_PICKLE), which broke
    the moment anyone swapped in their own DataFrame under a different
    filename -- e.g. `data/SAMPLE_100_BATCHES.pkl` instead of
    `data/SAMPLE_POEMS.pkl`. Resolution order, first match wins:

      1. `explicit` -- an exact path, e.g. from `--input` on the CLI.
      2. The `POETRY_DIACRITIZATION_INPUT` environment variable.
      3. Auto-detect: if exactly one `*.pkl` file sits in `data/` (ignoring
         this pipeline's own export outputs), use it.
      4. Fall back to the historical default, `data/SAMPLE_POEMS.pkl`.

    Raises FileNotFoundError with an actionable message if an explicit or
    env-var path doesn't exist, or if `data/` contains multiple candidate
    pickles and none of them is the default -- rather than silently
    guessing wrong.
    """
    if explicit:
        path = os.path.abspath(explicit)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--input path does not exist: {path}")
        return path

    env_path = os.environ.get(INPUT_PICKLE_ENV_VAR)
    if env_path:
        path = os.path.abspath(env_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"${INPUT_PICKLE_ENV_VAR} points to a file that doesn't exist: {path}"
            )
        return path

    data_dir = os.path.join(BASE_DIR, "data")
    export_names = {os.path.basename(os.path.join(BASE_DIR, "data", "diacritized_verses.pkl"))}
    candidates = (
        sorted(
            f
            for f in os.listdir(data_dir)
            if f.endswith(".pkl") and f not in export_names
        )
        if os.path.isdir(data_dir)
        else []
    )

    if len(candidates) == 1:
        return os.path.join(data_dir, candidates[0])

    if len(candidates) > 1:
        default_name = os.path.basename(INPUT_PICKLE)
        if default_name in candidates:
            return INPUT_PICKLE
        raise FileNotFoundError(
            f"Multiple .pkl files found in {data_dir} and none is the "
            f"default ({default_name}): {candidates}. Pass --input "
            "path/to/your.pkl (or set POETRY_DIACRITIZATION_INPUT) to "
            "disambiguate."
        )

    # Nothing in data/ at all -- fall back to the default so the resulting
    # FileNotFoundError (raised by pd.read_pickle) names a familiar path.
    return INPUT_PICKLE


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
# LLM provider
# ---------------------------------------------------------------------------
# Two providers are supported: DeepSeek's first-party API (the default,
# unchanged from before this setting existed), and NVIDIA's OpenAI-compatible
# NIM endpoint (integrate.api.nvidia.com), which currently serves DeepSeek
# models for free. Set the MODEL_PROVIDER env var to "nvidia" to switch;
# anything else (including unset) keeps the original DeepSeek behavior.
#
# This is read once into a plain constant rather than checked ad hoc all
# over the codebase, so every module agrees on which provider is active for
# the whole run.
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_NVIDIA = "nvidia"
SUPPORTED_PROVIDERS = [PROVIDER_DEEPSEEK, PROVIDER_NVIDIA]

MODEL_PROVIDER = (
    PROVIDER_NVIDIA if os.environ.get("MODEL_PROVIDER", "").strip().lower() == "nvidia"
    else PROVIDER_DEEPSEEK
)

# ---------------------------------------------------------------------------
# LLM / model settings
# ---------------------------------------------------------------------------
# DeepSeek's first-party API lineup (api-docs.deepseek.com), as of this
# writing. deepseek-v4-flash is the cheaper/faster default; deepseek-v4-pro
# is the stronger, more expensive one. Override per-run with `--model`.
DEEPSEEK_SUPPORTED_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"

# NVIDIA's NIM endpoint, currently hosting DeepSeek models under a different
# model-string naming scheme (the "deepseek-ai/" org prefix). Free for
# developers as of this writing -- see pricing.py/pricing_config.py, which
# already handle an unpriced model by leaving cost columns as None, so
# running on this provider needs no pricing setup at all.
NVIDIA_SUPPORTED_MODELS = ["deepseek-ai/deepseek-v4-flash"]
NVIDIA_DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"

# Provider-aware view used everywhere else in the codebase (cli.py's
# --model choices/default, llm_client.py's fallback when no model is passed
# in) so nothing needs its own if/else on MODEL_PROVIDER.
SUPPORTED_MODELS = (
    NVIDIA_SUPPORTED_MODELS if MODEL_PROVIDER == PROVIDER_NVIDIA else DEEPSEEK_SUPPORTED_MODELS
)
DEFAULT_MODEL = (
    NVIDIA_DEFAULT_MODEL if MODEL_PROVIDER == PROVIDER_NVIDIA else DEEPSEEK_DEFAULT_MODEL
)

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

# ---------------------------------------------------------------------------
# Checkpointing (git push)
# ---------------------------------------------------------------------------
# Long generation runs (e.g. on a Colab instance that can disappear without
# warning) commit + push runtime/ to the git remote every N completed
# batches, so progress survives even if the runtime dies mid-run. This
# assumes git authentication is already configured wherever the process
# runs (e.g. `git config` + a credential helper set up in the notebook,
# before calling into this pipeline) -- this pipeline never touches
# credentials itself. See git_checkpoint.py.
#
# Override per-run with `--checkpoint-every` / disable with `--no-checkpoint`.
CHECKPOINT_EVERY_N_BATCHES = 20
CHECKPOINT_ENABLED = True

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
