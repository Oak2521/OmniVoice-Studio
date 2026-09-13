"""Confucius4-TTS engine scaffold (#590).

The engine is opt-in (gated behind OMNIVOICE_CONFUCIUS4_TTS_DIR) and
subprocess-isolated, so it must be wired into the registry yet completely inert
on a default install — never importing the (unvalidated) upstream package, never
reporting available without a clone. These tests pin exactly that.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))


def test_registered_in_lazy_registry():
    from services.tts_backend import _LAZY_REGISTRY
    assert _LAZY_REGISTRY.get("confucius4-tts") == ("engines.confucius4", "Confucius4Backend")


def test_backend_class_metadata():
    from engines.confucius4 import Confucius4Backend
    assert Confucius4Backend.id == "confucius4-tts"
    assert Confucius4Backend.gpu_compat == ("cuda", "rocm", "xpu", "npu", "cpu")
    assert Confucius4Backend.supports_voice_design is False


def test_inert_without_clone_dir(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CONFUCIUS4_TTS_DIR", raising=False)
    from engines.confucius4 import bootstrap
    bootstrap.invalidate()
    assert bootstrap.is_confucius4_installed() is False

    from engines.confucius4 import Confucius4Backend
    ok, reason = Confucius4Backend.is_available()
    assert ok is False
    assert "OMNIVOICE_CONFUCIUS4_TTS_DIR" in reason


def test_resolve_raises_actionable_without_clone(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CONFUCIUS4_TTS_DIR", raising=False)
    from engines.confucius4 import bootstrap
    bootstrap.invalidate()
    import pytest
    with pytest.raises(RuntimeError, match="OMNIVOICE_CONFUCIUS4_TTS_DIR"):
        bootstrap.resolve_confucius4_venv()


def test_catalog_metadata_does_not_exclude_supported_accelerators():
    from engines.confucius4 import Confucius4Backend
    from services.tts_backend import _INSTALL_HINTS

    # The catalog label is device-neutral; routing metadata owns hardware claims.
    assert not re.search(
        r"\b(?:cuda|rocm|xpu|npu|cpu|mps)\b",
        Confucius4Backend.display_name,
        re.IGNORECASE,
    )
    assert "CUDA/CPU" not in _INSTALL_HINTS[Confucius4Backend.id]
    for family in Confucius4Backend.gpu_compat:
        assert family in _INSTALL_HINTS[Confucius4Backend.id].lower()
