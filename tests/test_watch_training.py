"""Read actual rotating-child logs without touching the training processes."""

import json
from pathlib import Path
import tempfile
import unittest

from morph_mimic.scripts.watch_training import render, snapshot


class TestTrainingWatcher(unittest.TestCase):
    def test_follows_new_child_and_distinguishes_live_iteration_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training.log").write_text("iter    990 | initial\n")
            status = dict(state="running", phase="waiting_for_existing_training", children=[],
                          current_iteration=None, maximum_iterations=20000,
                          chunk_iterations=1000, required_successes=3, evaluations=[])
            path = root / "supervisor_status.json"
            path.write_text(json.dumps(status))
            self.assertEqual(snapshot(root)["visible_iteration"], 990)
            child = root / "child2000.log"
            child.write_text("Isaac startup\niter   1040 | resumed\n")
            status.update(phase="train", current_iteration=1000, target_iteration=2000,
                          children=[dict(phase="train", log=str(child))])
            replacement = root / "replacement.json"
            replacement.write_text(json.dumps(status))
            replacement.replace(path)
            item = snapshot(root)
            self.assertEqual(item["completed_checkpoint_iteration"], 1000)
            self.assertEqual(item["visible_iteration"], 1040)
            self.assertEqual(item["active_log"], str(child))
            evaluation_log = root / "evaluate.log"
            evaluation_log.write_text("evaluating\n")
            status.update(phase="evaluate", current_iteration=2000)
            status["children"].append(dict(phase="evaluate", log=str(evaluation_log)))
            path.write_text(json.dumps(status))
            item = snapshot(root)
            self.assertEqual(item["visible_iteration"], 2000)
            self.assertEqual(item["active_log"], str(evaluation_log))
            self.assertEqual(item["last_training_line"], "iter   1040 | resumed")

    def test_stopping_states_explain_success_budget_and_failure_separately(self):
        base = dict(phase="finished", visible_iteration=20000, maximum_iterations=20000,
                    target_iteration=20000, consecutive_successes=0, required_successes=3,
                    evaluation_count=20, last_evaluation=None, last_training_line=None,
                    error="child exited 7", active_log="child.log")
        for state in ("validated", "budget_exhausted", "failed"):
            with self.subTest(state=state):
                text = "\n".join(render(dict(base, state=state)))
                self.assertIn(f"STOP {state}", text)
        self.assertIn("验收尚未通过", "\n".join(render(dict(base, state="budget_exhausted"))))
        self.assertIn("child exited 7", "\n".join(render(dict(base, state="failed"))))


if __name__ == "__main__":
    unittest.main()
