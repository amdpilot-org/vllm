# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITER FP4BMM gfx942 gating logic in envs.py.

These tests mock the ROCm platform so they can run on any hardware.
"""

import logging
import os
from unittest.mock import patch

import pytest

from vllm.envs import _resolve_fp4bmm_default

EXPECTED_LOG = (
    "Disabling VLLM_ROCM_USE_AITER_FP4BMM on gfx942: "
    "MXFP4 not supported by this hardware. Set =1 explicitly to override."
)


class TestAiterFp4BmmGating:
    """Gating logic for VLLM_ROCM_USE_AITER_FP4BMM on gfx942."""

    def test_default_disabled_on_gfx942(self):
        """When env var is NOT set and platform is gfx942, default must be False."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ROCM_USE_AITER_FP4BMM", None)
            with patch("vllm.platforms.rocm.on_gfx942", return_value=True):
                assert _resolve_fp4bmm_default() is False

    def test_explicit_override_respected_on_gfx942(self):
        """When env var is explicitly set to 1 on gfx942, user override is respected."""
        with patch.dict(os.environ, {"VLLM_ROCM_USE_AITER_FP4BMM": "1"}):
            with patch("vllm.platforms.rocm.on_gfx942", return_value=True):
                assert _resolve_fp4bmm_default() is True

    def test_explicit_override_zero_on_gfx942(self):
        """When env var is explicitly set to 0, it is respected even on gfx942."""
        with patch.dict(os.environ, {"VLLM_ROCM_USE_AITER_FP4BMM": "0"}):
            with patch("vllm.platforms.rocm.on_gfx942", return_value=True):
                assert _resolve_fp4bmm_default() is False

    def test_exact_log_string_emitted_on_gfx942(self, caplog):
        """The exact required warning must be logged when auto-disabling on gfx942."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ROCM_USE_AITER_FP4BMM", None)
            with patch("vllm.platforms.rocm.on_gfx942", return_value=True):
                with caplog.at_level(logging.WARNING, logger="vllm.envs"):
                    _resolve_fp4bmm_default()
        assert EXPECTED_LOG in caplog.text

    def test_non_gfx942_unaffected(self):
        """On non-gfx942 ROCm, the default remains enabled when env var is unset."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ROCM_USE_AITER_FP4BMM", None)
            with patch("vllm.platforms.rocm.on_gfx942", return_value=False):
                assert _resolve_fp4bmm_default() is True
