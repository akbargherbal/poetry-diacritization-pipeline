import os

import pytest

from poetry_diacritization import config


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Point config.BASE_DIR at a throwaway dir with its own data/ subfolder,
    so we can freely add/remove candidate .pkl files without touching the
    real project's data/ directory.
    """
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "INPUT_PICKLE", os.path.join(str(tmp_path), "data", "SAMPLE_POEMS.pkl"))
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.delenv(config.INPUT_PICKLE_ENV_VAR, raising=False)
    return tmp_path / "data"


def test_explicit_path_wins_even_if_it_does_not_look_like_data_dir(isolated_data_dir, tmp_path):
    explicit = tmp_path / "elsewhere.pkl"
    explicit.write_bytes(b"")
    assert config.resolve_input_pickle(str(explicit)) == str(explicit.resolve())


def test_explicit_path_missing_raises(isolated_data_dir):
    with pytest.raises(FileNotFoundError, match="--input"):
        config.resolve_input_pickle("/no/such/file.pkl")


def test_env_var_used_when_no_explicit_path(isolated_data_dir, tmp_path, monkeypatch):
    env_file = tmp_path / "from_env.pkl"
    env_file.write_bytes(b"")
    monkeypatch.setenv(config.INPUT_PICKLE_ENV_VAR, str(env_file))
    assert config.resolve_input_pickle() == str(env_file.resolve())


def test_env_var_missing_file_raises(isolated_data_dir, monkeypatch):
    monkeypatch.setenv(config.INPUT_PICKLE_ENV_VAR, "/no/such/file.pkl")
    with pytest.raises(FileNotFoundError, match=config.INPUT_PICKLE_ENV_VAR):
        config.resolve_input_pickle()


def test_single_pkl_in_data_dir_is_auto_detected(isolated_data_dir):
    my_file = isolated_data_dir / "SAMPLE_100_BATCHES.pkl"
    my_file.write_bytes(b"")
    assert config.resolve_input_pickle() == str(my_file)


def test_multiple_pkls_falls_back_to_default_if_present(isolated_data_dir):
    (isolated_data_dir / "SAMPLE_100_BATCHES.pkl").write_bytes(b"")
    default_file = isolated_data_dir / "SAMPLE_POEMS.pkl"
    default_file.write_bytes(b"")
    assert config.resolve_input_pickle() == config.INPUT_PICKLE


def test_multiple_pkls_without_default_raises(isolated_data_dir):
    (isolated_data_dir / "SAMPLE_100_BATCHES.pkl").write_bytes(b"")
    (isolated_data_dir / "OTHER_BATCHES.pkl").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Multiple .pkl files"):
        config.resolve_input_pickle()


def test_export_output_pickle_is_never_treated_as_a_candidate(isolated_data_dir):
    my_file = isolated_data_dir / "SAMPLE_100_BATCHES.pkl"
    my_file.write_bytes(b"")
    (isolated_data_dir / "diacritized_verses.pkl").write_bytes(b"")
    assert config.resolve_input_pickle() == str(my_file)


def test_empty_data_dir_falls_back_to_default(isolated_data_dir):
    assert config.resolve_input_pickle() == config.INPUT_PICKLE
