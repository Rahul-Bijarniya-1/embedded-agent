import os

from agent.config import BACKEND_DIR, Settings


def _base_env(monkeypatch):
    for var in (
        "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS",
        "GEMINI_MODEL", "GEMINI_THINKING_LEVEL", "GEMINI_TTS_MODEL", "GEMINI_TTS_LOCATION",
        "GEMINI_TTS_VOICE", "LLM_TIMEOUT_S", "TTS_TIMEOUT_S", "RATE_LIMIT_PER_MINUTE",
        "CACHE_TTL_S", "CACHE_SIZE", "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")


def test_settings_parses_model_lists_and_defaults(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("GEMINI_MODEL", " a-model , b-model ,,")
    monkeypatch.setenv("GEMINI_TTS_MODEL", "tts-a")
    s = Settings(_env_file=None)
    assert s.project == "proj"
    assert s.models == ["a-model", "b-model"]
    assert s.tts_models == ["tts-a"]
    assert s.location == "global"
    assert s.tts_location == "global"
    assert s.tts_voice == "Kore"
    assert s.thinking_level == "minimal"
    assert s.timeout_s == 30.0
    assert s.tts_timeout_s == 20.0
    assert s.rate_limit_per_minute == 20
    assert s.cache_ttl_s == 600
    assert s.cache_size == 256
    assert s.log_level == "INFO"


def test_settings_default_model_lists_are_ordered_fallbacks(monkeypatch):
    _base_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.models[0] == "gemini-3.5-flash"
    assert len(s.models) >= 2
    assert s.tts_models[0] == "gemini-3.8-flash-lite-tts"


def test_settings_empty_thinking_level_is_none(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "")
    assert Settings(_env_file=None).thinking_level is None


def test_settings_resolves_relative_credentials_path_against_backend_dir(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "secrets/sa.json")
    s = Settings(_env_file=None)
    assert s.credentials_path == str(BACKEND_DIR / "secrets" / "sa.json")


def test_settings_keeps_absolute_credentials_path(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/abs/key.json")
    assert Settings(_env_file=None).credentials_path == "/abs/key.json"


def test_export_credentials_sets_process_env(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "secrets/sa.json")
    s = Settings(_env_file=None)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
    s.export_credentials()
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == s.credentials_path


def test_export_credentials_without_path_leaves_env_alone(monkeypatch):
    _base_env(monkeypatch)
    s = Settings(_env_file=None)
    s.export_credentials()
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_settings_project_defaults_to_empty(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    assert Settings(_env_file=None).project == ""
