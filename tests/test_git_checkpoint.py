"""
Tests for git_checkpoint.py. No test here ever shells out to real git --
subprocess.run is monkeypatched throughout, since this suite is not allowed
to touch the real project's git history (see conftest.py's docstring).
"""
import subprocess

import pytest

from poetry_diacritization import config, git_checkpoint
from poetry_diacritization.git_checkpoint import CheckpointCounter, push_checkpoint


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run_factory(returncodes_by_subcommand: dict):
    """Build a fake subprocess.run that returns a chosen returncode based on
    the git subcommand (args[0][1], e.g. 'add'/'commit'/'push')."""

    def _fake_run(args, cwd=None, capture_output=None, text=None, timeout=None):
        subcommand = args[1]
        returncode = returncodes_by_subcommand.get(subcommand, 0)
        return _FakeCompletedProcess(returncode=returncode)

    return _fake_run


# ---------------------------------------------------------------------------
# push_checkpoint
# ---------------------------------------------------------------------------


def test_push_checkpoint_succeeds_when_add_commit_push_all_succeed(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory({}))
    assert push_checkpoint("checkpoint: 20 batches", cwd=str(tmp_path)) is True


def test_push_checkpoint_returns_false_when_add_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory({"add": 1}))
    assert push_checkpoint("checkpoint", cwd=str(tmp_path)) is False


def test_push_checkpoint_returns_false_when_nothing_to_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory({"commit": 1}))
    assert push_checkpoint("checkpoint", cwd=str(tmp_path)) is False


def test_push_checkpoint_returns_false_when_push_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory({"push": 1}))
    assert push_checkpoint("checkpoint", cwd=str(tmp_path)) is False


def test_push_checkpoint_never_raises_when_git_not_installed(monkeypatch, tmp_path):
    def _raise(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert push_checkpoint("checkpoint", cwd=str(tmp_path)) is False


def test_push_checkpoint_never_raises_on_timeout(monkeypatch, tmp_path):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git push", timeout=120)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert push_checkpoint("checkpoint", cwd=str(tmp_path)) is False


def test_push_checkpoint_never_raises_on_unexpected_error(monkeypatch, tmp_path):
    def _raise(*a, **k):
        raise RuntimeError("something weird")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert push_checkpoint("checkpoint", cwd=str(tmp_path)) is False


# ---------------------------------------------------------------------------
# CheckpointCounter
# ---------------------------------------------------------------------------


def test_checkpoint_counter_pushes_every_n_batches(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        git_checkpoint, "push_checkpoint", lambda message, cwd=None: pushed.append(message)
    )

    counter = CheckpointCounter(every_n=3, enabled=True)
    for _ in range(7):
        counter.batch_done()

    # Pushed at batch 3 and batch 6, not yet at 7.
    assert len(pushed) == 2


def test_checkpoint_counter_does_nothing_when_disabled(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        git_checkpoint, "push_checkpoint", lambda message, cwd=None: pushed.append(message)
    )

    counter = CheckpointCounter(every_n=1, enabled=False)
    for _ in range(5):
        counter.batch_done()

    assert pushed == []


def test_checkpoint_counter_defaults_come_from_config(monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_EVERY_N_BATCHES", 2)
    monkeypatch.setattr(config, "CHECKPOINT_ENABLED", True)
    pushed = []
    monkeypatch.setattr(
        git_checkpoint, "push_checkpoint", lambda message, cwd=None: pushed.append(message)
    )

    counter = CheckpointCounter()
    counter.batch_done()
    counter.batch_done()

    assert len(pushed) == 1


def test_checkpoint_counter_is_thread_safe_under_concurrent_batch_done(monkeypatch):
    import threading

    pushed = []
    monkeypatch.setattr(
        git_checkpoint, "push_checkpoint", lambda message, cwd=None: pushed.append(message)
    )

    counter = CheckpointCounter(every_n=10, enabled=True)
    threads = [threading.Thread(target=counter.batch_done) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(pushed) == 10
