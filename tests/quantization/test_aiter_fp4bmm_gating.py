# SPDX-License-Identifier: Apache-2.0
"""Test hardware-capability gating for VLLM_ROCM_USE_AITER_FP4BMM."""

import logging
import os
import unittest
from unittest import mock

import vllm.envs as envs


class TestAiterFp4BmmGating(unittest.TestCase):

    def tearDown(self):
        """Clear envs cache so each test sees fresh evaluation."""
        envs.disable_envs_cache()
        super().tearDown()

    def _check_default(self, gcn_arch: str, expected: bool):
        """Helper: mock torch.cuda.get_device_properties and read the env."""
        fake_props = mock.Mock()
        fake_props.gcnArchName = gcn_arch

        with mock.patch("torch.cuda.get_device_properties", return_value=fake_props), \
             mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.version.hip", True):
            envs.disable_envs_cache()
            val = envs.VLLM_ROCM_USE_AITER_FP4BMM
        self.assertEqual(val, expected, f"expected {expected} for {gcn_arch}")

    def test_gfx942_disabled(self):
        """MI300X/MI325X (gfx942) must auto-disable FP4BMM."""
        self._check_default("gfx942", False)

    def test_gfx950_enabled(self):
        """MI355X (gfx950) must keep FP4BMM enabled."""
        self._check_default("gfx950", True)

    def test_gfx90a_disabled(self):
        """Older gfx90a must also auto-disable."""
        self._check_default("gfx90a", False)

    def test_explicit_override_true(self):
        """User can force-enable on any arch."""
        os.environ["VLLM_ROCM_USE_AITER_FP4BMM"] = "1"
        try:
            self._check_default("gfx942", True)
        finally:
            del os.environ["VLLM_ROCM_USE_AITER_FP4BMM"]

    def test_explicit_override_false(self):
        """User can force-disable on any arch."""
        os.environ["VLLM_ROCM_USE_AITER_FP4BMM"] = "0"
        try:
            self._check_default("gfx950", False)
        finally:
            del os.environ["VLLM_ROCM_USE_AITER_FP4BMM"]

    def test_warning_logged_on_gfx942(self):
        """The expected one-line warning must be emitted."""
        fake_props = mock.Mock()
        fake_props.gcnArchName = "gfx942"
        with self.assertLogs("vllm.envs", level=logging.WARNING) as cm:
            with mock.patch("torch.cuda.get_device_properties", return_value=fake_props), \
                 mock.patch("torch.cuda.is_available", return_value=True), \
                 mock.patch("torch.version.hip", True):
                envs.disable_envs_cache()
                _ = envs.VLLM_ROCM_USE_AITER_FP4BMM
        self.assertTrue(
            any("Disabling VLLM_ROCM_USE_AITER_FP4BMM on gfx942" in msg
                for msg in cm.output),
            f"Expected warning not found in {cm.output}",
        )


if __name__ == "__main__":
    unittest.main()
