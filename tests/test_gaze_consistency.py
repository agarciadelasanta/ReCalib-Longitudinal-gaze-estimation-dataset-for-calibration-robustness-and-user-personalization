import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from statistical_analysis import ConsistencyConfig, analyze_consistency_frame
from visualization.explore_gaze_consistency import build_html, read_report


def _vector(pitch_deg, yaw_deg):
    pitch, yaw = np.radians(pitch_deg), np.radians(yaw_deg)
    return -np.cos(pitch) * np.sin(yaw), -np.sin(pitch), -np.cos(pitch) * np.cos(yaw)


def _frame(*, eth_follows_head=True, seed=0):
    """Two users x three sessions x four targets x ten frames.

    The ReCalib angle is target + session drift + head motion. The ETH angle adds a
    constant offset and either the same drift and head motion or independent noise.
    """

    rng = np.random.default_rng(seed)
    targets = [(-10.0, -15.0), (-10.0, 15.0), (10.0, -15.0), (10.0, 15.0)]
    rows = []
    for user in ("00", "01"):
        for session in ("00", "01", "02"):
            drift = rng.normal(0.0, 1.0, size=2)
            for index, (target_pitch, target_yaw) in enumerate(targets):
                for frame in range(10):
                    head = rng.normal(0.0, 0.5, size=2)
                    recalib = np.array([target_pitch, target_yaw]) + drift + head
                    variation = drift + head if eth_follows_head else rng.normal(0.0, 1.0, size=2)
                    eth = np.array([target_pitch, target_yaw]) + variation + [2.0, 0.5]
                    row = {
                        "user_id": user, "session_id": session, "task_type": "test",
                        "discarded": False, "eth_ok": 1.0, "eth_face_center_z": 600.0,
                        "pog_px_x": float(index % 2), "pog_px_y": float(index // 2),
                    }
                    row.update(zip(("gaze_vector_x", "gaze_vector_y", "gaze_vector_z"), _vector(*recalib)))
                    row.update(zip(("eth_label_x", "eth_label_y", "eth_label_z"), _vector(*eth)))
                    rows.append(row)
    return pd.DataFrame(rows)


def _summary(result, level, view, component):
    summary = result.summary
    return summary[
        summary["level"].eq(level) & summary["view"].eq(view)
        & summary["task_type"].eq("all") & summary["component"].eq(component)
    ].iloc[0]


class GazeConsistencyTests(unittest.TestCase):
    def test_vectors_sharing_variation_agree_at_every_level(self):
        result = analyze_consistency_frame(
            _frame(), ConsistencyConfig(min_session_samples=0, bootstrap_iterations=20)
        )

        for level, view in (
            ("within_session", "raw"),
            ("within_session", "target_conditioned"),
            ("across_sessions", "target_conditioned"),
        ):
            for component in ("pitch", "yaw"):
                row = _summary(result, level, view, component)
                self.assertGreater(row["pearson_r"], 0.99, (level, view, component))
                self.assertAlmostEqual(row["sd_ratio"], 1.0, delta=0.05)
                self.assertLessEqual(row["pearson_r_ci_low"], row["pearson_r"])
        offsets = result.offsets[result.offsets["task_type"].eq("all")]
        self.assertTrue(np.allclose(offsets["mean_offset_pitch_deg"], 2.0, atol=0.1))
        self.assertEqual(len(result.session_deviations[result.session_deviations["task_type"].eq("all")]), 6)

    def test_independent_head_variation_breaks_target_conditioned_agreement(self):
        result = analyze_consistency_frame(
            _frame(eth_follows_head=False),
            ConsistencyConfig(min_session_samples=0, bootstrap_iterations=0),
        )

        self.assertGreater(_summary(result, "within_session", "raw", "yaw")["pearson_r"], 0.95)
        conditioned = _summary(result, "within_session", "target_conditioned", "yaw")
        self.assertLess(abs(conditioned["pearson_r"]), 0.3)
        self.assertGreater(conditioned["sd_ratio"], 1.5)
        self.assertTrue(np.isnan(conditioned["pearson_r_ci_low"]))

    def test_discarded_failed_and_small_sessions_are_excluded(self):
        frame = _frame()
        frame.loc[0, "discarded"] = True
        frame.loc[1, "eth_ok"] = 0.0
        frame.loc[2, "eth_face_center_z"] = 2500.0  # false dlib detection, null in the release
        extra = frame.iloc[[5, 6]].assign(session_id="09")
        frame = pd.concat([frame, extra], ignore_index=True)

        result = analyze_consistency_frame(
            frame, ConsistencyConfig(min_session_samples=10, bootstrap_iterations=0)
        )

        self.assertEqual(result.metadata["discarded_rows_excluded"], 1)
        self.assertEqual(result.metadata["eth_unavailable_rows_excluded"], 2)
        self.assertEqual(result.metadata["small_sessions_excluded"], 1)
        self.assertEqual(result.metadata["accepted_rows"], len(frame) - 5)
        self.assertNotIn("09", set(result.offsets["session_id"]))

    def test_missing_columns_and_bad_config_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "eth_label_x"):
            analyze_consistency_frame(_frame().drop(columns="eth_label_x"))
        with self.assertRaisesRegex(ValueError, "two distinct"):
            ConsistencyConfig(vectors=("recalib", "recalib"))

    def test_cli_writes_report_and_explorer(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "summary.csv"
            output = Path(directory) / "report"
            _frame().to_csv(source, index=False)
            subprocess.run(
                [
                    sys.executable, "-m", "statistical_analysis", "consistency", str(source),
                    "--output", str(output), "--bootstrap", "5", "--min-session-samples", "0",
                    "--no-progress",
                ],
                cwd=repository, check=True, capture_output=True, text=True,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["report_type"], "gaze_consistency")

            report = read_report(output)
            document = build_html(report, output.name)
            self.assertIn("Gaze vector consistency", document)
            self.assertIn("ETH-XGaze label", document)
            self.assertNotIn("__SUMMARY__", document)


if __name__ == "__main__":
    unittest.main()
