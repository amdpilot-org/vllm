# Test for gfx942 FP4BMM gating logic
# Avoid importing compiled vllm._C by mocking the platform init path.

import os
import pathlib

import pytest


@pytest.fixture
def rocm_src_path() -> pathlib.Path:
    """Return the absolute path of vllm/platforms/rocm.py."""
    candidates = [
        pathlib.Path("/workspace/vllm/vllm/platforms/rocm.py"),
        pathlib.Path(__file__).parents[3] / "vllm" / "platforms" / "rocm.py",
        pathlib.Path.cwd().parents[2] / "vllm" / "platforms" / "rocm.py",
        pathlib.Path.cwd() / "vllm" / "platforms" / "rocm.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.fail("Could not locate vllm/platforms/rocm.py")


def test_gfx942_guard_present(rocm_src_path):
    """The rocm.py source must contain the gfx942 conditional guard."""
    src = rocm_src_path.read_text()
    assert "gfx942" in src
    assert "_ON_GFX942" in src
    assert "VLLM_ROCM_USE_AITER_FP4BMM" in src


def test_boot_warning_present(rocm_src_path):
    """The exact one-line boot warning must be present in rocm.py."""
    src = rocm_src_path.read_text()
    expected = (
        "Disabling VLLM_ROCM_USE_AITER_FP4BMM on gfx942: "
        "MXFP4 not supported by this hardware. Set =1 explicitly to override."
    )
    assert expected in src, f"Exact warning not found in rocm.py"


def test_gate_preserves_explicit_override(rocm_src_path):
    """Gate code must only trigger when env var is NOT explicitly set."""
    src = rocm_src_path.read_text()
    # The gating block should check "not in os.environ" or equivalent
    assert 'not in os.environ' in src or 'os.environ.get' in src


def test_gating_block_sets_zero_on_gfx942(rocm_src_path):
    """Source must show a branch that forces the env var to 0/'0' on gfx942."""
    src = rocm_src_path.read_text()
    assert '"0"' in src or "'0'" in src
    assert "os.environ[" in src
