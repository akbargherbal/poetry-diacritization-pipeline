"""
Periodic "checkpoint" commits + pushes of runtime/ during a long generation
run, so progress survives an interrupted/killed process (e.g. a Colab
runtime disconnecting mid-run).

This module deliberately does NOT touch git authentication or
configuration at all -- it assumes `git` is already set up (remote,
credentials, user.name/user.email) in whatever environment this runs,
exactly as agreed: auth is configured in the notebook, outside this
project's code.

Every function here is best-effort and non-fatal: a checkpoint failing
(nothing to commit, network hiccup, auth problem, git not installed) is
logged as a warning and the generation run continues uninterrupted. An
expensive LLM pass should never crash because of a git problem -- the
whole point of this feature is to reduce data loss risk, not introduce a
new way to lose an otherwise-successful run.
"""
import logging
import subprocess

from . import config

log = logging.getLogger("poetry_diacritization")


def _run_git(args: list, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def push_checkpoint(message: str, cwd: str = None) -> bool:
    """
    Stage runtime/ (registry.pkl + raw_responses/), commit with `message`,
    and push. Returns True if a commit was actually made and pushed, False
    on any failure or if there was nothing new to commit -- never raises.
    """
    cwd = cwd or config.BASE_DIR

    try:
        add_result = _run_git(["add", "runtime/"], cwd=cwd)
        if add_result.returncode != 0:
            log.warning(f"Checkpoint skipped: `git add` failed: {add_result.stderr.strip()}")
            return False

        commit_result = _run_git(["commit", "-m", message], cwd=cwd)
        if commit_result.returncode != 0:
            # Very often just "nothing to commit" (no new completed
            # batches since the last checkpoint) -- not an error worth
            # alarming over, but still surfaced at debug level.
            log.debug(f"Checkpoint: nothing to commit ({commit_result.stdout.strip()})")
            return False

        push_result = _run_git(["push"], cwd=cwd)
        if push_result.returncode != 0:
            log.warning(f"Checkpoint committed locally but push failed: {push_result.stderr.strip()}")
            return False

        log.info(f"Checkpoint pushed: {message}")
        return True

    except FileNotFoundError:
        log.warning("Checkpoint skipped: git is not installed / not on PATH.")
        return False
    except subprocess.TimeoutExpired:
        log.warning("Checkpoint skipped: a git command timed out.")
        return False
    except Exception as e:
        log.warning(f"Checkpoint skipped due to an unexpected error: {e}")
        return False


class CheckpointCounter:
    """
    Tracks completed batches across a generation pass (which may run
    batches concurrently across threads) and triggers a checkpoint push
    every `every_n` batches. Thread-safe.
    """

    def __init__(self, every_n: int = None, enabled: bool = None, cwd: str = None):
        self.every_n = every_n if every_n is not None else config.CHECKPOINT_EVERY_N_BATCHES
        self.enabled = enabled if enabled is not None else config.CHECKPOINT_ENABLED
        self.cwd = cwd or config.BASE_DIR
        self._count = 0
        import threading

        self._lock = threading.Lock()

    def batch_done(self):
        """Call once per successfully completed batch. Pushes a checkpoint
        when the count crosses a multiple of `every_n`."""
        if not self.enabled or self.every_n <= 0:
            return

        with self._lock:
            self._count += 1
            should_push = self._count % self.every_n == 0
            count_at_push = self._count

        if should_push:
            push_checkpoint(
                message=f"checkpoint: {count_at_push} batches generated this run",
                cwd=self.cwd,
            )
