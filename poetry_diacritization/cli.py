import argparse
import logging

from . import config
from .export import export_passed
from .generate import run_generation_pass
from .registry import load_or_build_registry, status_counts
from .threshold import apply_threshold, score_distribution
from .validate import reset_for_revalidation, run_validation_pass


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def cmd_status(args):
    df = load_or_build_registry(getattr(args, "input", None))
    print("\nStatus counts:")
    print(status_counts(df).to_string())
    dist = score_distribution(df)
    if dist is not None:
        print("\npyarud score distribution (all verses ever scored):")
        print(dist.to_string())
    print(f"\nRegistry: {len(df)} verses total, at {config.REGISTRY_PATH}")


def cmd_generate(args):
    df = load_or_build_registry(getattr(args, "input", None))
    run_generation_pass(
        df,
        model=args.model,
        thinking_enabled=args.thinking,
        reasoning_effort=args.reasoning_effort,
        save_reasoning=args.save_reasoning,
    )
    cmd_status(args)


def cmd_validate(args):
    df = load_or_build_registry(getattr(args, "input", None))
    run_validation_pass(df)
    cmd_status(args)


def cmd_revalidate(args):
    df = load_or_build_registry(getattr(args, "input", None))
    df, reset_ids, skipped_ids = reset_for_revalidation(df, statuses=tuple(args.statuses))
    if not reset_ids:
        print(f"Nothing to revalidate (no rows in {list(args.statuses)} with a saved raw response).")
        if skipped_ids:
            print(f"{len(skipped_ids)} row(s) in that status have no saved raw response, skipped: {skipped_ids}")
        return
    print(f"Revalidating {len(reset_ids)} verse(s), offline (no LLM calls): {reset_ids}")
    run_validation_pass(df)
    cmd_status(args)


def cmd_threshold(args):
    df = load_or_build_registry(getattr(args, "input", None))
    apply_threshold(df, cutoff=args.cutoff, include_rescored=args.include_rescored)
    cmd_status(args)


def cmd_all(args):
    df = load_or_build_registry(getattr(args, "input", None))
    run_generation_pass(
        df,
        model=args.model,
        thinking_enabled=args.thinking,
        reasoning_effort=args.reasoning_effort,
        save_reasoning=args.save_reasoning,
    )
    run_validation_pass(df)
    apply_threshold(df, cutoff=args.cutoff, include_rescored=args.include_rescored)
    cmd_status(args)


def cmd_export(args):
    df = load_or_build_registry(getattr(args, "input", None))
    export_passed(df)


def main():
    _setup_logging()
    parser = argparse.ArgumentParser(prog="poetry_diacritization")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_input_arg(p):
        p.add_argument(
            "--input",
            dest="input",
            default=None,
            help=(
                "Path to your batch-level input pickle (a DataFrame with "
                "poem_no/meter/DATA columns), replacing data/SAMPLE_POEMS.pkl. "
                "Only used the first time a run builds runtime/registry.pkl; "
                "ignored once that registry already exists. If omitted, falls "
                f"back to ${config.INPUT_PICKLE_ENV_VAR} if set, then to the "
                "only *.pkl found in data/, then to data/SAMPLE_POEMS.pkl."
            ),
        )

    def _add_llm_args(p):
        p.add_argument(
            "--model",
            choices=config.SUPPORTED_MODELS,
            default=config.DEFAULT_MODEL,
            help=f"Which DeepSeek model to call (default: {config.DEFAULT_MODEL})",
        )
        think_group = p.add_mutually_exclusive_group()
        think_group.add_argument(
            "--thinking",
            dest="thinking",
            action="store_true",
            default=None,
            help="Enable thinking/reasoning mode",
        )
        think_group.add_argument(
            "--no-thinking",
            dest="thinking",
            action="store_false",
            help="Disable thinking/reasoning mode (default)",
        )
        p.add_argument(
            "--reasoning-effort",
            choices=config.SUPPORTED_REASONING_EFFORTS,
            default=config.DEFAULT_REASONING_EFFORT,
            help="Only used if --thinking is on. 'low'/'medium' aren't offered because "
            "DeepSeek's own API collapses them to 'high' anyway.",
        )
        save_reasoning_group = p.add_mutually_exclusive_group()
        save_reasoning_group.add_argument(
            "--save-reasoning",
            dest="save_reasoning",
            action="store_true",
            default=config.SAVE_REASONING_ARTIFACTS,
            help="Persist the model's reasoning/thinking trace to "
            "runtime/raw_responses/<call_id>_reasoning.txt. Only meaningful "
            "when --thinking is on; ignored otherwise.",
        )
        save_reasoning_group.add_argument(
            "--no-save-reasoning",
            dest="save_reasoning",
            action="store_false",
            help=f"Do not persist reasoning traces (default: "
            f"{'off' if not config.SAVE_REASONING_ARTIFACTS else 'on'}).",
        )

    p_generate = sub.add_parser("generate", help="Call the LLM for everything still pending")
    _add_llm_args(p_generate)
    _add_input_arg(p_generate)
    p_generate.set_defaults(func=cmd_generate)

    p_validate = sub.add_parser("validate", help="Parse + check fidelity + score with pyarud, offline")
    _add_input_arg(p_validate)
    p_validate.set_defaults(func=cmd_validate)

    p_revalidate = sub.add_parser(
        "revalidate",
        help="Re-check failed_text_mismatch/failed_parse verses against current code, "
        "offline (no LLM calls) — for after editing normalize() or the parser",
    )
    p_revalidate.add_argument(
        "--statuses",
        nargs="+",
        default=["failed_text_mismatch", "failed_parse"],
        help="Which status(es) to reset and re-check (default: %(default)s)",
    )
    _add_input_arg(p_revalidate)
    p_revalidate.set_defaults(func=cmd_revalidate)

    p_threshold = sub.add_parser("threshold", help="Turn stored pyarud scores into passed/failed")
    p_threshold.add_argument("--cutoff", type=float, default=0.90)
    p_threshold.add_argument("--include-rescored", action="store_true")
    _add_input_arg(p_threshold)
    p_threshold.set_defaults(func=cmd_threshold)

    p_all = sub.add_parser("all", help="Run generate -> validate -> threshold in one go")
    _add_llm_args(p_all)
    p_all.add_argument("--cutoff", type=float, default=0.90)
    p_all.add_argument("--include-rescored", action="store_true")
    _add_input_arg(p_all)
    p_all.set_defaults(func=cmd_all)

    p_status = sub.add_parser("status", help="Print status counts and score distribution")
    _add_input_arg(p_status)
    p_status.set_defaults(func=cmd_status)

    p_export = sub.add_parser("export", help="Write the clean passed-only dataset to data/")
    _add_input_arg(p_export)
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
