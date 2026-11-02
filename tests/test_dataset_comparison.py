import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from statistical_analysis import AnalysisConfig, compare_csvs, compare_frames
from visualization.explore_dataset_comparison import build_html, read_report


def _row(dataset, user, session, target_x, value, *, discarded=False):
    return {
        "image": f"{dataset}_{user}_{session}_{target_x}_{value}.png",
        "sample_id": f"{target_x}_{value}",
        "task_id": "00",
        "session_id": session,
        "user_id": user,
        "task_type": "9-point",
        "pog_px_x": float(target_x),
        "pog_px_y": 100.0,
        "head_rot_x": float(value),
        "head_rot_y": 0.0,
        "head_rot_z": 0.0,
        "discarded": discarded,
    }


class DatasetComparisonTests(unittest.TestCase):
    def test_independent_datasets_return_kpi_vector_and_exclude_discarded(self):
        first = pd.DataFrame(
            [
                _row("a", "00", "00", 100.0, 0.0),
                _row("a", "00", "01", 100.0, 1.0),
                _row("a", "00", "01", 100.0, 10000.0, discarded=True),
            ]
        )
        second = pd.DataFrame(
            [
                _row("b", "00", "00", 100.0, 5.0),
                _row("b", "00", "01", 100.0, 6.0),
            ]
        )
        second["task_type"] = "free-viewing"

        result = compare_frames(
            first,
            second,
            AnalysisConfig(
                feature_families=("head_rotation",),
                views=("raw",),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
            label_a="first",
            label_b="second",
        )

        self.assertEqual(
            set(result.dataset_distances["metric"]),
            {
                "location_shift",
                "dispersion_shift",
                "distribution_shift",
                "dependence_shift",
            },
        )
        self.assertGreater(
            result.dataset_distances.query("metric == 'location_shift'")["estimate"].item(),
            0.0,
        )
        self.assertEqual(result.metadata["dataset_a"]["discarded_rows_excluded"], 1)
        self.assertEqual(result.metadata["row_pairing"], "none_independent_datasets")
        self.assertEqual(
            result.metadata["task_stratification"],
            "none_all_accepted_samples_combined",
        )
        self.assertTrue(
            result.dataset_distances["analysis_scope"]
            .eq("all_accepted_samples")
            .all()
        )

    def test_repeated_frames_do_not_change_equal_session_comparison(self):
        first = pd.DataFrame(
            [
                _row("a", "00", "00", 100.0, 0.0),
                _row("a", "00", "01", 100.0, 10.0),
            ]
        )
        repeated = pd.concat(
            [first, pd.DataFrame([first.iloc[0].to_dict()] * 20)],
            ignore_index=True,
        )
        second = pd.DataFrame(
            [
                _row("b", "20", "00", 100.0, 2.0),
                _row("b", "20", "01", 100.0, 4.0),
            ]
        )
        config = AnalysisConfig(
            feature_families=("head_rotation",),
            views=("raw",),
            bootstrap_iterations=0,
            permutation_iterations=0,
        )

        base = compare_frames(first, second, config).dataset_distances
        duplicated = compare_frames(repeated, second, config).dataset_distances
        pd.testing.assert_series_equal(base["estimate"], duplicated["estimate"])

    def test_users_are_equal_weight_even_when_session_counts_differ(self):
        first = pd.DataFrame(
            [
                _row("a", "00", "00", 100.0, 0.0),
                _row("a", "01", "00", 100.0, 10.0),
            ]
        )
        with_extra_sessions = pd.concat(
            [
                first,
                pd.DataFrame(
                    [
                        _row("a", "00", "01", 100.0, 0.0),
                        _row("a", "00", "02", 100.0, 0.0),
                        _row("a", "00", "03", 100.0, 0.0),
                    ]
                ),
            ],
            ignore_index=True,
        )
        second = pd.DataFrame(
            [
                _row("b", "10", "00", 100.0, 3.0),
                _row("b", "11", "00", 100.0, 5.0),
            ]
        )
        config = AnalysisConfig(
            feature_families=("head_rotation",),
            views=("raw",),
            bootstrap_iterations=0,
            permutation_iterations=0,
        )

        base = compare_frames(first, second, config).dataset_distances
        extra = compare_frames(with_extra_sessions, second, config).dataset_distances
        pd.testing.assert_series_equal(base["estimate"], extra["estimate"])

    def test_target_conditioning_removes_target_composition_difference(self):
        rows_a = []
        rows_b = []
        for index in range(9):
            rows_a.append(_row("a", "00", "00", 0.0, 0.0))
            rows_b.append(_row("b", "10", "00", 2736.0, 10.0))
        rows_a.append(_row("a", "00", "00", 2736.0, 10.0))
        rows_b.append(_row("b", "10", "00", 0.0, 0.0))

        result = compare_frames(
            pd.DataFrame(rows_a),
            pd.DataFrame(rows_b),
            AnalysisConfig(
                feature_families=("head_rotation",),
                views=("raw", "target_conditioned"),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
        )
        locations = result.dataset_distances.query(
            "metric == 'location_shift'"
        ).set_index("view")["estimate"]
        self.assertGreater(locations["raw"], 0.0)
        self.assertAlmostEqual(locations["target_conditioned"], 0.0)
        conditioned = result.dataset_distances.query("view == 'target_conditioned'")
        self.assertTrue((conditioned["n_common_targets"] == 2).all())

    def test_session_resampling_produces_intervals_and_adjusted_p_values(self):
        rows_a = []
        rows_b = []
        for session_number in range(4):
            for value in (0.0, 1.0, 2.0):
                first = _row("a", "00", str(session_number), 100.0, value + session_number)
                second = _row("b", "10", str(session_number), 100.0, value + session_number + 4.0)
                first["head_rot_y"] = value
                second["head_rot_y"] = -value
                rows_a.append(first)
                rows_b.append(second)

        result = compare_frames(
            pd.DataFrame(rows_a),
            pd.DataFrame(rows_b),
            AnalysisConfig(
                feature_families=("head_rotation",),
                views=("raw",),
                bootstrap_iterations=8,
                permutation_iterations=8,
                random_seed=9,
            ),
        ).dataset_distances

        self.assertTrue(result["ci_low"].notna().all())
        self.assertTrue(result["ci_high"].notna().all())
        self.assertTrue(result["p_value"].between(0.0, 1.0).all())
        self.assertTrue(result["q_value"].between(0.0, 1.0).all())
        self.assertTrue((result["q_value"] >= result["p_value"]).all())

    def test_task_headers_are_optional_for_overall_comparison(self):
        first = pd.DataFrame([_row("a", "00", "00", 100.0, 0.0)]).drop(
            columns=["task_id", "task_type"]
        )
        second = pd.DataFrame([_row("b", "10", "00", 100.0, 2.0)]).drop(
            columns=["task_id", "task_type"]
        )
        result = compare_frames(
            first,
            second,
            AnalysisConfig(
                feature_families=("head_rotation",),
                views=("raw",),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
        )
        self.assertEqual(len(result.dataset_distances), 4)

    def test_unused_audit_headers_may_differ_between_datasets(self):
        first = pd.DataFrame([_row("a", "00", "00", 100.0, 0.0)])
        second = pd.DataFrame([_row("b", "10", "00", 100.0, 2.0)])
        first["discard_info"] = "accepted"
        second["discard_reason"] = "accepted"
        first["qa_max_x"] = 0.5
        second["qa_max_x"] = 0.4

        result = compare_frames(
            first,
            second,
            AnalysisConfig(
                feature_families=("head_rotation",),
                views=("raw",),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
        )

        audit = result.column_compatibility.set_index("column")
        self.assertNotIn("discard_info", audit.index)
        self.assertNotIn("discard_reason", audit.index)
        self.assertFalse(any(column.startswith("qa_") for column in audit.index))

    def test_csv_report_cli_and_schema_validation(self):
        first = pd.DataFrame(
            [_row("a", "00", session, 100.0, value) for session, value in (("00", 0), ("01", 1))]
        )
        second = pd.DataFrame(
            [_row("b", "10", session, 100.0, value) for session, value in (("00", 3), ("01", 4))]
        )
        with self.assertRaisesRegex(ValueError, "Required comparison columns"):
            compare_frames(first, second.drop(columns=["head_rot_z"]))

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path_a = root / "first.csv"
            path_b = root / "second.csv"
            output = root / "comparison"
            first.to_csv(path_a, index=False)
            second.to_csv(path_b, index=False)

            manifest = compare_csvs(
                path_a,
                path_b,
                output,
                AnalysisConfig(
                    feature_families=("head_rotation",),
                    views=("raw",),
                    bootstrap_iterations=0,
                    permutation_iterations=0,
                ),
            )
            self.assertEqual(
                set(manifest.files),
                {
                    "manifest.json",
                    "dataset_distances.csv",
                    "comparison_coverage.csv",
                    "column_compatibility.csv",
                    "feature_catalog.csv",
                },
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "statistical_analysis",
                    "compare",
                    str(path_a),
                    str(path_b),
                    "--output",
                    str(root / "cli-comparison"),
                    "--families",
                    "head_rotation",
                    "--views",
                    "raw",
                    "--bootstrap",
                    "0",
                    "--permutations",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["report_type"], "independent_dataset_comparison")

            report = read_report(output)
            self.assertEqual(report["manifest"]["metadata"]["dataset_a"]["sessions_per_user"],
                             {"min": 2, "median": 2.0, "max": 2, "mean": 2.0})
            self.assertEqual(report["manifest"]["metadata"]["dataset_a"]["samples_per_session"]["mean"], 1.0)
            explorer = build_html(report, output.name)
            self.assertIn("Data used in the comparison", explorer)
            self.assertIn("Feature completeness", explorer)
            self.assertIn("Recording balance", explorer)
            self.assertIn("0.00%", explorer)
            self.assertNotIn("__DATA_USED__", explorer)
            for dataset in ("dataset_a", "dataset_b"):
                report["manifest"]["metadata"][dataset].pop("sessions_per_user")
                report["manifest"]["metadata"][dataset].pop("samples_per_session")
            self.assertIn("Not recorded", build_html(report, output.name))
            self.assertIn("Independent Dataset Domain-Gap Explorer", explorer)
            self.assertIn("location-chart", explorer)
            self.assertIn('"metric":"location_shift"', explorer)
            self.assertNotIn("__DISTANCES__", explorer)
            self.assertNotIn(
                'label(statistic)+" %{x:.4f}<br>Estimate', explorer
            )
            self.assertNotIn("95% CI %{customdata[1]", explorer)
            self.assertIn(
                "95% confidence intervals and p/q values are unavailable",
                explorer,
            )
            self.assertIn('aria-label="Explain location shift"', explorer)
            self.assertIn('aria-label="Explain dependence shift"', explorer)
            self.assertIn("no universal anomaly cutoff", explorer)
            self.assertIn("q &lt; 0.05", explorer)


if __name__ == "__main__":
    unittest.main()
