import argparse
import logging

from . import config
from .export import export_passed
from .generate import run_generation_pass
from .registry import load_or_build_registry, status_counts
from .threshold import apply_threshold, score_distribution
from .validate import run_validation_pass


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
    df = load_or_build_registry()
    print("\nStatus counts:")
    print(status_counts(df).to_string())
    dist = score_distribution(df)
    if dist is not None:
        print("\npyarud score distribution (all verses ever scored):")
        print(dist.to_string())
    print(f"\nRegistry: {len(df)} verses total, at {config.REGISTRY_PATH}")


def cmd_generate(args):
    df = load_or_build_registry()
    run_generation_pass(df)
    cmd_status(args)


def cmd_validate(args):
    df = load_or_build_registry()
    run_validation_pass(df)
    cmd_status(args)


def cmd_threshold(args):
    df = load_or_build_registry()
    apply_threshold(df, cutoff=args.cutoff, include_rescored=args.include_rescored)
    cmd_status(args)


def cmd_all(args):
    df = load_or_build_registry()
    run_generation_pass(df)
    run_validation_pass(df)
    apply_threshold(df, cutoff=args.cutoff, include_rescored=args.include_rescored)
    cmd_status(args)


def cmd_export(args):
    df = load_or_build_registry()
    export_passed(df)


def main():
    _setup_logging()
    parser = argparse.ArgumentParser(prog="poetry_diacritization")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="Call the LLM for everything still pending").set_defaults(
        func=cmd_generate
    )
    sub.add_parser("validate", help="Parse + check fidelity + score with pyarud, offline").set_defaults(
        func=cmd_validate
    )

    p_threshold = sub.add_parser("threshold", help="Turn stored pyarud scores into passed/failed")
    p_threshold.add_argument("--cutoff", type=float, default=0.90)
    p_threshold.add_argument("--include-rescored", action="store_true")
    p_threshold.set_defaults(func=cmd_threshold)

    p_all = sub.add_parser("all", help="Run generate -> validate -> threshold in one go")
    p_all.add_argument("--cutoff", type=float, default=0.90)
    p_all.add_argument("--include-rescored", action="store_true")
    p_all.set_defaults(func=cmd_all)

    sub.add_parser("status", help="Print status counts and score distribution").set_defaults(
        func=cmd_status
    )
    sub.add_parser("export", help="Write the clean passed-only dataset to data/").set_defaults(
        func=cmd_export
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
