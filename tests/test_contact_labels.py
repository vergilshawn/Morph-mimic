"""Reject corrupt or incomplete reference-contact supervision."""

import unittest

import numpy as np

from morph_mimic.utils.motion import validate_contact_labels


class TestContactLabelValidation(unittest.TestCase):
    def labels(self, frames=20):
        return dict(
            contact_flags=np.zeros((frames, 9), dtype=bool),
            contact_targets=np.zeros((frames, 9, 3), dtype=np.float32),
            contact_target_flags=np.zeros((frames, 9), dtype=bool),
            fps=np.array([50.0]),
            source_num_frames=np.array(frames),
            source_init_phase=np.array(0.0),
            source_end_phase=np.array(1.0),
        )

    def test_complete_replay(self):
        validate_contact_labels(self.labels(), 20, 50.0)

    def test_failed_replay_cannot_stretch_over_source_motion(self):
        with self.assertRaisesRegex(ValueError, "complete motion"):
            validate_contact_labels(self.labels(10), 20, 50.0)

    def test_wrong_sampling_frequency(self):
        labels = self.labels()
        labels["fps"] = np.array([30.0])
        with self.assertRaisesRegex(ValueError, "fps"):
            validate_contact_labels(labels, 20, 50.0)

    def test_nonfinite_contact_target(self):
        labels = self.labels()
        labels["contact_targets"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_contact_labels(labels, 20, 50.0)

    def test_nonbinary_occupancy(self):
        labels = self.labels()
        labels["contact_target_flags"] = np.full((20, 9), 0.5)
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_contact_labels(labels, 20, 50.0)

    def test_full_sized_record_with_wrong_phase_is_rejected(self):
        labels = self.labels()
        labels["source_end_phase"] = np.array(0.8)
        with self.assertRaisesRegex(ValueError, "ended before"):
            validate_contact_labels(labels, 20, 50.0)


if __name__ == "__main__":
    unittest.main()
