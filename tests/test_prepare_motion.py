"""Check that reference preparation preserves frame alignment and source data."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from morph_mimic.scripts.prepare_motion import prepare_clip


class TestPrepareMotion(unittest.TestCase):
    def source(self, root):
        path = Path(root) / "walk1_subject1.npz"
        arrays = dict(fps=np.array([10.0]), joint_names=np.array(["joint"]), body_names=np.array(["base"]))
        for key, width in (("joint_pos", 1), ("joint_vel", 1), ("body_pos_w", 3), ("body_quat_w", 4),
                           ("body_lin_vel_w", 3), ("body_ang_vel_w", 3), ("extra_per_frame", 2)):
            arrays[key] = np.arange(21 * width, dtype=np.float32).reshape(21, width)
        np.savez(path, **arrays)
        return path

    def test_all_time_arrays_share_the_same_window_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            original = hashlib.sha256(source.read_bytes()).hexdigest()
            output, meta = prepare_clip(source, Path(directory) / "prepared", 0.5, 0.8)
            self.assertEqual(output.name, source.name)
            self.assertEqual(meta["prepared_num_frames"], 9)
            with np.load(output) as result, np.load(source) as full:
                for key in meta["sliced_fields"]:
                    np.testing.assert_array_equal(result[key], full[key][5:14])
                np.testing.assert_array_equal(result["joint_names"], full["joint_names"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)
            with self.assertRaises(FileExistsError):
                prepare_clip(source, output.parent, 0.5, 0.8)

    def test_invalid_range_and_non_frame_aligned_requests_leave_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            output = Path(directory) / "prepared"
            for start, duration in ((1.5, 1.0), (0.05, 0.5), (0.0, -1.0)):
                with self.subTest(start=start, duration=duration), self.assertRaises(ValueError):
                    prepare_clip(source, output, start, duration)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
