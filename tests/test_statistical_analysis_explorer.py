import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from visualization.explore_statistical_analysis import build_html, read_report


class StatisticalAnalysisExplorerTests(unittest.TestCase):
    def test_explorer_embeds_report_tables_and_interactive_plots(self):
        with TemporaryDirectory() as temporary_directory:
            report_dir = Path(temporary_directory)
            (report_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "accepted_rows": 40,
                            "users": 2,
                            "user_sessions": 4,
                        }
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "view": "raw",
                        "task_type": "9-point",
                        "family": "head_rotation",
                        "metric": "distribution_shift",
                        "metric_method": "robust_scaled_energy_distance",
                        "user_a": "00",
                        "user_b": "01",
                        "estimate": 1.25,
                        "ci_low": 1.0,
                        "ci_high": 1.5,
                        "p_value": 0.02,
                        "q_value": 0.04,
                        "n_sessions_a": 2,
                        "n_sessions_b": 2,
                        "n_samples_a": 20,
                        "n_samples_b": 20,
                    }
                ]
            ).to_csv(report_dir / "pairwise_distances.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": "00",
                        "task_type": "9-point",
                        "view": "raw",
                        "family": "head_rotation",
                        "metric": "longitudinal_instability_median",
                        "estimate": 0.2,
                        "n_sessions": 2,
                        "n_samples": 20,
                    }
                ]
            ).to_csv(report_dir / "user_profiles.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": "00",
                        "session_id": "00",
                        "family": "head_rotation",
                        "metric": "distribution_shift",
                        "aggregation": "session",
                        "estimate": 0.8,
                        "ci_low": None,
                        "ci_high": None,
                        "n_sessions": 1,
                    }
                ]
            ).to_csv(report_dir / "calibration_test_mismatch.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "family": "head_rotation",
                        "dimensions": 3,
                        "supported_views": "raw",
                        "notes": "Head rotation.",
                    }
                ]
            ).to_csv(report_dir / "feature_catalog.csv", index=False)

            report = read_report(report_dir)
            explorer = build_html(report, report_dir.name)

            self.assertIn("ReCalib Statistical Analysis Explorer", explorer)
            self.assertIn('"estimate":1.25', explorer)
            self.assertIn("matrix-chart", explorer)
            self.assertIn("ranking-chart", explorer)
            self.assertIn("profile-chart", explorer)
            self.assertIn("mismatch-chart", explorer)
            self.assertNotIn("__PAIRWISE_JSON__", explorer)


if __name__ == "__main__":
    unittest.main()
