import os
import sys
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops


EXPECTED_WARNING = (
    "Disabling VLLM_ROCM_USE_AITER_FP4BMM on gfx942: "
    "MXFP4 not supported by this hardware. Set =1 explicitly to override."
)


class TestAiterFp4BmmGating:
    """Validate gfx942-aware gating for VLLM_ROCM_USE_AITER_FP4BMM."""

    @pytest.fixture(autouse=True)
    def _clean_env(self):
        """Remove the env var before every test."""
        os.environ.pop("VLLM_ROCM_USE_AITER_FP4BMM", None)
        envs.disable_envs_cache()
        yield
        os.environ.pop("VLLM_ROCM_USE_AITER_FP4BMM", None)
        envs.disable_envs_cache()

    def test_gfx942_unset_disables_and_warns(self):
        """gfx942 + unset env var -> disabled and warning emitted."""
        with patch("vllm.envs.logger.warning") as mock_warning:
            with patch.object(
                sys.modules["torch.cuda"],
                "is_available",
                return_value=True,
            ):
                fake_props = type("Props", (), {"gcnArchName": "gfx942"})()
                with patch.object(
                    sys.modules["torch.cuda"],
                    "get_device_properties",
                    return_value=fake_props,
                ):
                    result = envs.VLLM_ROCM_USE_AITER_FP4BMM
        assert result is False
        mock_warning.assert_called_once_with(EXPECTED_WARNING)

    def test_gfx942_explicit_one_enabled(self):
        """gfx942 + explicit =1 -> enabled, no warning."""
        os.environ["VLLM_ROCM_USE_AITER_FP4BMM"] = "1"
        with patch("vllm.envs.logger.warning") as mock_warning:
            with patch.object(
                sys.modules["torch.cuda"],
                "is_available",
                return_value=True,
            ):
                fake_props = type("Props", (), {"gcnArchName": "gfx942"})()
                with patch.object(
                    sys.modules["torch.cuda"],
                    "get_device_properties",
                    return_value=fake_props,
                ):
                    result = envs.VLLM_ROCM_USE_AITER_FP4BMM
        assert result is True
        mock_warning.assert_not_called()

    def test_non_gfx942_defaults_enabled(self):
        """non-gfx942 + unset -> enabled, no warning."""
        with patch("vllm.envs.logger.warning") as mock_warning:
            with patch.object(
                sys.modules["torch.cuda"],
                "is_available",
                return_value=True,
            ):
                fake_props = type("Props", (), {"gcnArchName": "gfx950"})()
                with patch.object(
                    sys.modules["torch.cuda"],
                    "get_device_properties",
                    return_value=fake_props,
                ):
                    result = envs.VLLM_ROCM_USE_AITER_FP4BMM
        assert result is True
        mock_warning.assert_not_called()
