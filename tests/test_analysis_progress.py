import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from statistical_analysis.__main__ import main
from statistical_analysis.progress import progress, progress_enabled


class AnalysisProgressTests(unittest.TestCase):
    def test_cli_progress_preserves_reports_for_both_commands(self):
        frame = pd.DataFrame([
            {
                "user_id": str(user), "session_id": str(session),
                "task_type": "9-point", "discarded": False,
                "head_rot_x": user + session + sample / 10,
                "head_rot_y": sample / 20, "head_rot_z": session / 10,
            }
            for user in range(2)
            for session in range(2)
            for sample in range(3)
        ])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            csv_a, csv_b = root / "a.csv", root / "b.csv"
            frame.to_csv(csv_a, index=False)
            frame.assign(head_rot_x=frame.head_rot_x + 1).to_csv(csv_b, index=False)
            for command in ([str(csv_a)], ["compare", str(csv_a), str(csv_b)]):
                with self.subTest(command=command):
                    outputs = []
                    for quiet in (False, True):
                        output = root / ("quiet" if quiet else "progress")
                        stdout, stderr = io.StringIO(), io.StringIO()
                        args = [*command, "--output", str(output), "--families",
                                "head_rotation", "--views", "raw", "--bootstrap",
                                "2", "--permutations", "2"]
                        if quiet:
                            args.append("--no-progress")
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            self.assertEqual(main(args), 0)
                        json.loads(stdout.getvalue())
                        if quiet:
                            self.assertEqual(stderr.getvalue(), "")
                        else:
                            self.assertIn("Bootstrap" if command[0] != "compare" else "Dataset bootstrap", stderr.getvalue())
                            self.assertIn("Permutations", stderr.getvalue())
                        outputs.append({
                            path.relative_to(output): path.read_bytes()
                            for path in output.rglob("*.csv")
                        })
                    self.assertEqual(outputs[0], outputs[1])

    def test_progress_scope_resets_after_error(self):
        with self.assertRaises(ValueError):
            with progress_enabled():
                raise ValueError("interrupted analysis")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(list(progress(range(2), "Library call")), [0, 1])
        self.assertEqual(stderr.getvalue(), "")
