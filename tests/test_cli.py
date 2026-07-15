import pytest

from poetry_diacritization import cli, config


@pytest.fixture(autouse=True)
def no_real_logging_setup(monkeypatch):
    # Avoid FileHandler lifecycle issues across tests / repeated main() calls;
    # logging setup itself isn't under test here.
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)


@pytest.fixture
def stub_pipeline_funcs(monkeypatch, make_registry_df):
    """Stub out every function main() can dispatch to, and record calls."""
    calls = {}
    df = make_registry_df([{"verse_id": "1_001", "status": "pending"}])

    monkeypatch.setattr(cli, "load_or_build_registry", lambda: df)

    def _record(name):
        def _fn(*args, **kwargs):
            calls[name] = (args, kwargs)
            return df

        return _fn

    monkeypatch.setattr(cli, "run_generation_pass", _record("run_generation_pass"))
    monkeypatch.setattr(cli, "run_validation_pass", _record("run_validation_pass"))
    monkeypatch.setattr(cli, "apply_threshold", _record("apply_threshold"))
    monkeypatch.setattr(cli, "export_passed", _record("export_passed"))
    monkeypatch.setattr(cli, "status_counts", lambda df: df["status"].value_counts())
    monkeypatch.setattr(cli, "score_distribution", lambda df: None)

    return calls


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["poetry_diacritization"] + argv)
    cli.main()


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def test_status_command_dispatches(monkeypatch, stub_pipeline_funcs, capsys):
    _run_cli(monkeypatch, ["status"])
    out = capsys.readouterr().out
    assert "Status counts" in out


def test_generate_command_dispatches_with_defaults(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["generate"])
    args, kwargs = stub_pipeline_funcs["run_generation_pass"]
    assert kwargs["model"] == config.DEFAULT_MODEL
    assert kwargs["thinking_enabled"] is None  # resolved later inside run_generation_pass
    assert kwargs["reasoning_effort"] == config.DEFAULT_REASONING_EFFORT
    assert kwargs["save_reasoning"] == config.SAVE_REASONING_ARTIFACTS


def test_validate_command_dispatches(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["validate"])
    assert "run_validation_pass" in stub_pipeline_funcs


def test_threshold_command_passes_cutoff_and_include_rescored(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["threshold", "--cutoff", "0.85", "--include-rescored"])
    args, kwargs = stub_pipeline_funcs["apply_threshold"]
    assert kwargs["cutoff"] == 0.85
    assert kwargs["include_rescored"] is True


def test_threshold_command_default_cutoff(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["threshold"])
    args, kwargs = stub_pipeline_funcs["apply_threshold"]
    assert kwargs["cutoff"] == 0.90
    assert kwargs["include_rescored"] is False


def test_all_command_runs_generate_validate_threshold(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["all"])
    assert "run_generation_pass" in stub_pipeline_funcs
    assert "run_validation_pass" in stub_pipeline_funcs
    assert "apply_threshold" in stub_pipeline_funcs


def test_export_command_dispatches(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["export"])
    assert "export_passed" in stub_pipeline_funcs


def test_no_command_raises_system_exit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["poetry_diacritization"])
    with pytest.raises(SystemExit):
        cli.main()


# ---------------------------------------------------------------------------
# --thinking / --no-thinking (mutually exclusive)
# ---------------------------------------------------------------------------


def test_thinking_flag_enables_thinking(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["generate", "--thinking"])
    _, kwargs = stub_pipeline_funcs["run_generation_pass"]
    assert kwargs["thinking_enabled"] is True


def test_no_thinking_flag_disables_thinking(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["generate", "--no-thinking"])
    _, kwargs = stub_pipeline_funcs["run_generation_pass"]
    assert kwargs["thinking_enabled"] is False


def test_thinking_and_no_thinking_together_is_a_system_exit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["poetry_diacritization", "generate", "--thinking", "--no-thinking"])
    with pytest.raises(SystemExit):
        cli.main()


# ---------------------------------------------------------------------------
# --save-reasoning / --no-save-reasoning (mutually exclusive, defaults to config)
# ---------------------------------------------------------------------------


def test_save_reasoning_flag(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["generate", "--save-reasoning"])
    _, kwargs = stub_pipeline_funcs["run_generation_pass"]
    assert kwargs["save_reasoning"] is True


def test_no_save_reasoning_flag(monkeypatch, stub_pipeline_funcs):
    _run_cli(monkeypatch, ["generate", "--no-save-reasoning"])
    _, kwargs = stub_pipeline_funcs["run_generation_pass"]
    assert kwargs["save_reasoning"] is False


def test_save_reasoning_defaults_to_config_value(monkeypatch, stub_pipeline_funcs):
    monkeypatch.setattr(config, "SAVE_REASONING_ARTIFACTS", False)
    _run_cli(monkeypatch, ["generate"])
    _, kwargs = stub_pipeline_funcs["run_generation_pass"]
    assert kwargs["save_reasoning"] is False


def test_save_reasoning_and_no_save_reasoning_together_is_a_system_exit(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["poetry_diacritization", "generate", "--save-reasoning", "--no-save-reasoning"]
    )
    with pytest.raises(SystemExit):
        cli.main()


# ---------------------------------------------------------------------------
# --reasoning-effort / --model choices
# ---------------------------------------------------------------------------


def test_reasoning_effort_invalid_choice_is_system_exit(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["poetry_diacritization", "generate", "--reasoning-effort", "medium"]
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_model_invalid_choice_is_system_exit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["poetry_diacritization", "generate", "--model", "gpt-4"])
    with pytest.raises(SystemExit):
        cli.main()


def test_model_flag_passed_through(monkeypatch, stub_pipeline_funcs):
    model = config.SUPPORTED_MODELS[0]
    _run_cli(monkeypatch, ["generate", "--model", model])
    _, kwargs = stub_pipeline_funcs["run_generation_pass"]
    assert kwargs["model"] == model
