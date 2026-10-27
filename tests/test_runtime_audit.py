"""Check the inertia-frame conversion used by the real Isaac runtime audit."""

import unittest

import numpy as np

from morph_mimic.scripts.audit_runtime_reference import inertia_in_link_frame


class TestRuntimeAudit(unittest.TestCase):
    def test_principal_tensor_rotates_into_link_axes_with_quaternion_sign_invariance(self):
        # A 45-degree rotation of principal moments 1,3 gives nonzero Ixy.
        quaternion = np.array([np.cos(np.pi/8), 0., 0., np.sin(np.pi/8)])
        expected = np.array([[2., -1., 0.], [-1., 2., 0.], [0., 0., 5.]])
        np.testing.assert_allclose(inertia_in_link_frame([1., 3., 5.], quaternion), expected, atol=1e-12)
        np.testing.assert_allclose(inertia_in_link_frame([1., 3., 5.], -quaternion), expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
