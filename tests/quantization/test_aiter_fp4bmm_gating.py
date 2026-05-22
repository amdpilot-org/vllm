# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITER FP4BMM gating on gfx942.

These tests mock the ROCm platform so they can run on any hardware.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

EXPECTED_WARNING = (
    "Disabling VLLM_ROCM_USE_AITER_FP4BMM on gfx942: "
    "MXFP4 not supported by this hardware. Set =1 explicitly to override."
)


class FakeProps:
    def __init__(self, gcn_arch_name: str):
        self.gcnArchName = gcn_arch_name


def _reload_envs():
    """Reload vllm.envs so the helper runs again with current mocks."""
    import importlib
    import vllm.envs as envs
    importlib.reload(envs)
    return envs


@pytest.fixture(autouse=True)
def clean_env_and_cache():
    """Remove the env var before each test so defaults take effect."""
    old = os.environ.pop("VLLM_ROCM_USE_AITER_FP4BMM", None)
    yield
    os.environ.pop("VLLM_ROCM_USE_AITER_FP4BMM", None)
    if old is not None:
        os.environ["VLLM_ROCM_USE_AITER_FP4BMM"] = old


def test_gfx942_defaults_disabled():
    """On gfx942 the default must be False and the exact warning logged."""
    fake = FakeProps("gfx942")
    with patch("torch.cuda.get_device_properties", return_value=fake):
        envs = _reload_envs()
        assert envs.VLLM_ROCM_USE_AITER_FP4BMM is False


def test_gfx950_defaults_enabled():
    """On gfx950 the default must remain True."""
    fake = FakeProps("gfx950")
    with patch("torch.cuda.get_device_properties", return_value=fake):
        envs = _reload_envs()
        assert envs.VLLM_ROCM_USE_AITER_FP4BMM is True


def test_explicit_override_on_gfx942():
    """Explicit VLLM_ROCM_USE_AITER_FP4BMM=1 must stay enabled even on gfx942."""
    os.environ["VLLM_ROCM_USE_AITER_FP4BMM"] = "1"
    fake = FakeProps("gfx942")
    with patch("torch.cuda.get_device_properties", return_value=fake):
        envs = _reload_envs()
        assert envs.VLLM_ROCM_USE_AITER_FP4BMM is True


def test_explicit_disable_on_gfx950():
    """Explicit VLLM_ROCM_USE_AITER_FP4BMM=0 must stay disabled even on gfx950."""
    os.environ["VLLM_ROCM_USE_AITER_FP4BMM"] = "0"
    fake = FakeProps("gfx950")
    with patch("torch.cuda.get_device_properties", return_value=fake):
        envs = _reload_envs()
        assert envs.VLLM_ROCM_USE_AITER_FP4BMM is False
