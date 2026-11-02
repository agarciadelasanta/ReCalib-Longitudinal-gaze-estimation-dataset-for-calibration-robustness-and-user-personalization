import unittest
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pandas as pd

from statistical_analysis import AnalysisConfig, analyze_csv, analyze_frame


LANDMARK_IDS = (
    4, 33, 61, 129, 133, 168, 263, 308, 358,
    362, 469, 470, 471, 472, 474, 475, 476, 477,
)


def _row(user, session, task_type, value, *, discarded=False):
    return {
        "image": f"{user}_{session}_{task_type}_{value}.png",
        "sample_id": str(value),
        "task_id": "00" if task_type == "9-point" else "01",
        "session_id": session,
        "user_id": user,
        "task_type": task_type,
        "pog_px_x": 100.0,
        "pog_px_y": 100.0,
        "head_rot_x": float(value),
        "head_rot_y": 0.0,
        "head_rot_z": 0.0,
        "discarded": discarded,
        "img_res_width": 640,
        "img_res_height": 480,
    }


def _rich_row(user, session, task_type, value):
    row = _row(user, session, task_type, value)
    row.update(
        gaze_vector_x=0.01 * value,
        gaze_vector_y=0.02 * value,
        gaze_vector_z=-1.0,
        gaze_origin_x=value,
        gaze_origin_y=value + 1.0,
        gaze_origin_z=400.0 + value,
        gaze_pog_x=value,
        gaze_pog_y=value + 2.0,
        gaze_pog_z=0.0,
        head_trans_x=value,
        head_trans_y=value + 1.0,
        head_trans_z=400.0 + value,
        qa_max_x=0.5,
        qa_min_x=0.4,
        qa_max_y=0.6,
        qa_min_y=0.3,
        qa_mean_x=0.45,
        qa_mean_y=0.45,
        qa_head_bbox_surface=0.08,
        qa_ear_left=0.25,
        qa_ear_right=0.25,
    )
    for index, landmark_id in enumerate(LANDMARK_IDS):
        row[f"lm_{landmark_id}_x"] = 500.0 + value + index
        row[f"lm_{landmark_id}_y"] = 400.0 + value + index / 2.0
    return row


class AnalyzeFrameTests(unittest.TestCase):
    def test_missing_feature_values_skip_only_affected_task_family(self):
        rows = [
            _rich_row(user, "00", task, value)
            for task in ("9-point", "16-point")
            for user, value in (("00", 1), ("01", 2))
        ]
        frame = pd.DataFrame(rows)
        frame.loc[frame.task_type == "16-point", "qa_ear_left"] = float("nan")
        result = analyze_frame(frame, AnalysisConfig(
            feature_families=("head_rotation", "quality"),
            bootstrap_iterations=0, permutation_iterations=0,
        ))
        pairs = result.pairwise_distances
        self.assertEqual(set(pairs[pairs.family == "head_rotation"].task_type),
                         {"9-point", "16-point"})
        self.assertEqual(set(pairs[pairs.family == "quality"].task_type), {"9-point"})
        self.assertEqual(set(pairs.view), {"raw", "target_conditioned"})

    def test_configuration_rejects_invalid_inference_settings(self):
        with self.assertRaisesRegex(ValueError, "confidence_level"):
            AnalysisConfig(confidence_level=1.5)
        with self.assertRaisesRegex(ValueError, "bootstrap_iterations"):
            AnalysisConfig(bootstrap_iterations=-1)

    def test_analysis_excludes_discarded_rows_and_weights_sessions_equally(self):
        rows = []
        for task_type in ("9-point", "16-point"):
            rows.extend(
                [
                    _row("00", "00", task_type, 0.0),
                    _row("00", "01", task_type, 10.0),
                    _row("01", "00", task_type, 2.0),
                    _row("01", "01", task_type, 4.0),
                    _row("00", "00", task_type, 10_000.0, discarded=True),
                ]
            )

        base = pd.DataFrame(rows)
        repeated = pd.concat(
            [
                base,
                pd.concat(
                    [base[(base.user_id == "00") & (base.session_id == "00") & ~base.discarded]]
                    * 25,
                    ignore_index=True,
                ),
            ],
            ignore_index=True,
        )
        config = AnalysisConfig(
            feature_families=("head_rotation",),
            views=("raw",),
            bootstrap_iterations=0,
            permutation_iterations=0,
        )

        first = analyze_frame(base, config)
        second = analyze_frame(repeated, config)

        self.assertEqual(first.metadata["discarded_rows_excluded"], 2)
        self.assertEqual(
            set(first.pairwise_distances["task_type"]),
            {"9-point", "16-point"},
        )
        first_location = first.pairwise_distances.query(
            "family == 'head_rotation' and metric == 'location_shift'"
        )["estimate"].tolist()
        second_location = second.pairwise_distances.query(
            "family == 'head_rotation' and metric == 'location_shift'"
        )["estimate"].tolist()
        self.assertEqual(first_location, second_location)

    def test_target_conditioning_removes_target_composition_shift(self):
        rows = []
        for user, first_count, second_count in (("00", 9, 1), ("01", 1, 9)):
            for index in range(first_count):
                row = _row(user, "00", "9-point", 0.0)
                row.update(pog_px_x=0.0, pog_px_y=0.0, image=f"{user}_a_{index}.png")
                rows.append(row)
            for index in range(second_count):
                row = _row(user, "00", "9-point", 10.0)
                row.update(pog_px_x=2736.0, pog_px_y=1824.0, image=f"{user}_b_{index}.png")
                rows.append(row)

        result = analyze_frame(
            pd.DataFrame(rows),
            AnalysisConfig(
                feature_families=("head_rotation",),
                views=("raw", "target_conditioned"),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
        )

        values = result.pairwise_distances[
            result.pairwise_distances.metric.eq("location_shift")
        ].set_index("view")["estimate"]
        self.assertGreater(values["raw"], 0.0)
        self.assertAlmostEqual(values["target_conditioned"], 0.0)

    def test_gaze_direction_uses_angular_geometry_not_vector_magnitude(self):
        rows = []
        for user, magnitude in (("00", 1.0), ("01", 3.0)):
            for session in ("00", "01"):
                row = _row(user, session, "9-point", 0.0)
                row.update(
                    gaze_vector_x=0.0,
                    gaze_vector_y=0.0,
                    gaze_vector_z=-magnitude,
                )
                rows.append(row)

        result = analyze_frame(
            pd.DataFrame(rows),
            AnalysisConfig(
                feature_families=("gaze_direction",),
                views=("raw",),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
        )

        estimate = result.pairwise_distances.loc[
            result.pairwise_distances.metric.eq("location_shift"), "estimate"
        ].item()
        self.assertAlmostEqual(estimate, 0.0)

    def test_core_kpi_vector_is_exposed_for_each_user_pair(self):
        rows = []
        for user, offset, multiplier in (("00", 0.0, 1.0), ("01", 2.0, 2.0)):
            for session in ("00", "01", "02"):
                for value in (0.0, 1.0, 2.0, 4.0):
                    row = _row(user, session, "9-point", offset + multiplier * value)
                    direction = 1.0 if user == "00" else -1.0
                    row["head_rot_y"] = direction * multiplier * value
                    row["head_rot_z"] = (
                        -value if user == "00" else multiplier * value**2
                    )
                    rows.append(row)

        result = analyze_frame(
            pd.DataFrame(rows),
            AnalysisConfig(
                feature_families=("head_rotation",),
                views=("raw",),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
        )

        self.assertEqual(
            set(result.pairwise_distances["metric"]),
            {
                "location_shift",
                "dispersion_shift",
                "distribution_shift",
                "dependence_shift",
            },
        )
        self.assertTrue((result.pairwise_distances["estimate"] > 0).all())

    def test_landmarks_separate_position_scale_and_normalized_shape(self):
        rows = []
        base_points = [(float(i % 6), float(i // 6)) for i in range(len(LANDMARK_IDS))]
        for user, offset, scale in (("00", 0.0, 1.0), ("01", 100.0, 2.0)):
            for session in ("00", "01"):
                row = _row(user, session, "9-point", 0.0)
                for landmark_id, (x, y) in zip(LANDMARK_IDS, base_points):
                    row[f"lm_{landmark_id}_x"] = offset + scale * x
                    row[f"lm_{landmark_id}_y"] = offset + scale * y
                rows.append(row)

        result = analyze_frame(
            pd.DataFrame(rows),
            AnalysisConfig(
                feature_families=(
                    "landmark_position",
                    "landmark_scale",
                    "landmark_shape",
                ),
                views=("raw",),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
        )
        location = result.pairwise_distances.query(
            "metric == 'location_shift'"
        ).set_index("family")["estimate"]

        self.assertGreater(location["landmark_position"], 0.0)
        self.assertGreater(location["landmark_scale"], 0.0)
        self.assertAlmostEqual(location["landmark_shape"], 0.0)

    def test_resampling_outputs_are_finite_adjusted_and_reproducible(self):
        rows = []
        for user, offset in (("00", 0.0), ("01", 5.0), ("02", 10.0)):
            for session_number in range(4):
                for value in (0.0, 1.0, 2.0, 3.0):
                    row = _row(
                        user,
                        f"{session_number:02d}",
                        "9-point",
                        value + offset + session_number,
                    )
                    row["head_rot_y"] = value * (int(user) + 1)
                    row["head_rot_z"] = value**2 + offset
                    rows.append(row)
        config = AnalysisConfig(
            feature_families=("head_rotation",),
            views=("raw",),
            bootstrap_iterations=20,
            permutation_iterations=20,
            random_seed=7,
        )

        first = analyze_frame(pd.DataFrame(rows), config).pairwise_distances
        second = analyze_frame(pd.DataFrame(rows), config).pairwise_distances

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["ci_low"].notna().all())
        self.assertTrue(first["ci_high"].notna().all())
        self.assertTrue((first["ci_low"] <= first["ci_high"]).all())
        self.assertTrue(first["p_value"].between(0.0, 1.0).all())
        self.assertTrue(first["q_value"].between(0.0, 1.0).all())
        self.assertTrue((first["q_value"] >= first["p_value"]).all())

    def test_default_analysis_excludes_landmark_families(self):
        rows = [
            _rich_row(user, session, task_type, value)
            for user, value in (("00", 0.0), ("01", 2.0))
            for session in ("00", "01")
            for task_type in ("9-point", "16-point")
        ]
        result = analyze_frame(
            pd.DataFrame(rows),
            AnalysisConfig(bootstrap_iterations=0, permutation_iterations=0),
        )

        expected = {
            "target_coverage",
            "gaze_direction",
            "gaze_origin",
            "gaze_point",
            "head_rotation",
            "head_translation",
            "quality",
        }
        self.assertEqual(set(result.feature_catalog["family"]), expected)
        self.assertEqual(set(result.pairwise_distances["family"]), expected)
        distribution_methods = result.pairwise_distances.query(
            "metric == 'distribution_shift'"
        ).groupby("family")["metric_method"].first()
        self.assertEqual(
            distribution_methods["head_rotation"],
            "robust_scaled_energy_distance",
        )
        gaze_pog_z = result.column_catalog.set_index("column").loc["gaze_pog_z"]
        self.assertEqual(gaze_pog_z["role"], "unusable")
        self.assertEqual(gaze_pog_z["reason"], "constant_on_accepted_rows")

    def test_profiles_capture_session_drift_and_calibration_test_mismatch(self):
        rows = []
        for user in ("00", "01"):
            for session_number in range(3):
                session_shift = 0.0 if user == "00" else session_number * 3.0
                for task_type in ("9-point", "16-point"):
                    task_shift = 5.0 if user == "01" and task_type == "16-point" else 0.0
                    for value in (0.0, 1.0, 2.0):
                        rows.append(
                            _row(
                                user,
                                f"{session_number:02d}",
                                task_type,
                                value + session_shift + task_shift,
                            )
                        )
        result = analyze_frame(
            pd.DataFrame(rows),
            AnalysisConfig(
                feature_families=("head_rotation",),
                views=("raw",),
                bootstrap_iterations=0,
                permutation_iterations=0,
            ),
        )

        drift = result.user_profiles.query(
            "task_type == '9-point' and view == 'raw' and "
            "family == 'head_rotation' and metric == 'longitudinal_instability_median'"
        ).set_index("user_id")["estimate"]
        self.assertAlmostEqual(drift["00"], 0.0)
        self.assertGreater(drift["01"], drift["00"])

        mismatch = result.calibration_test_mismatch.query(
            "aggregation == 'user_equal_session_mean' and "
            "family == 'head_rotation' and metric == 'location_shift'"
        ).set_index("user_id")["estimate"]
        self.assertAlmostEqual(mismatch["00"], 0.0)
        self.assertGreater(mismatch["01"], mismatch["00"])

    def test_csv_report_writes_long_tables_manifest_and_symmetric_matrices(self):
        rows = [
            _row(user, session, task_type, value)
            for user, value in (("00", 0.0), ("01", 2.0))
            for session in ("00", "01")
            for task_type in ("9-point", "16-point")
        ]
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.csv"
            output_path = root / "report"
            pd.DataFrame(rows).to_csv(input_path, index=False)

            manifest = analyze_csv(
                input_path,
                output_path,
                AnalysisConfig(
                    feature_families=("head_rotation",),
                    views=("raw",),
                    bootstrap_iterations=0,
                    permutation_iterations=0,
                ),
            )

            self.assertEqual(manifest.metadata["users"], 2)
            self.assertEqual(manifest.metadata["user_sessions"], 4)

            expected_files = {
                "manifest.json",
                "pairwise_distances.csv",
                "user_profiles.csv",
                "calibration_test_mismatch.csv",
                "feature_catalog.csv",
                "column_catalog.csv",
                "matrix_index.csv",
            }
            self.assertTrue(expected_files.issubset(set(manifest.files)))
            self.assertTrue(all((output_path / name).exists() for name in expected_files))
            index = pd.read_csv(output_path / "matrix_index.csv")
            self.assertEqual(
                set(index["statistic"]),
                {"estimate", "ci_low", "ci_high", "p_value", "q_value"},
            )
            estimate_path = index[index["statistic"].eq("estimate")].iloc[0]["path"]
            matrix = pd.read_csv(
                output_path / estimate_path,
                index_col=0,
                dtype={"user_id": "string"},
            )
            self.assertEqual(matrix.loc["00", "01"], matrix.loc["01", "00"])
            self.assertEqual(matrix.loc["00", "00"], 0.0)

    def test_command_line_runs_the_csv_report_seam(self):
        rows = [
            _row(user, session, task_type, value)
            for user, value in (("00", 0.0), ("01", 2.0))
            for session in ("00", "01")
            for task_type in ("9-point", "16-point")
        ]
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.csv"
            output_path = root / "report"
            pd.DataFrame(rows).to_csv(input_path, index=False)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "statistical_analysis",
                    str(input_path),
                    "--output",
                    str(output_path),
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
            self.assertTrue((output_path / "manifest.json").exists())


def _eth_frame():
    """Rich rows whose ETH columns copy the released labels; one extra row per task fails ETH."""

    rows = []
    for task in ("9-point", "16-point"):
        for user, session, value in (
            ("00", "00", 1), ("00", "01", 3), ("01", "00", 2), ("01", "01", 5),
        ):
            for offset in range(3):
                row = _rich_row(user, session, task, value + offset)
                row.update(
                    eth_ok=1,
                    eth_label_x=row["gaze_vector_x"],
                    eth_label_y=row["gaze_vector_y"],
                    eth_label_z=row["gaze_vector_z"],
                    eth_face_center_x=row["gaze_origin_x"],
                    eth_face_center_y=row["gaze_origin_y"],
                    eth_face_center_z=row["gaze_origin_z"],
                )
                rows.append(row)
        failed = _rich_row("00", "00", task, 1_000)
        failed.update(eth_ok=0)
        rows.append(failed)
    return pd.DataFrame(rows)


class ETHLabelSetTests(unittest.TestCase):
    def test_eth_labels_reproduce_recalib_results_on_identical_values(self):
        frame = _eth_frame()
        families = ("gaze_direction", "gaze_origin", "gaze_point")
        settings = dict(views=("raw",), bootstrap_iterations=0, permutation_iterations=0)
        recalib = analyze_frame(
            frame[frame.eth_ok.eq(1)],
            AnalysisConfig(feature_families=families, **settings),
        )
        eth = analyze_frame(
            frame, AnalysisConfig(feature_families=families, label_set="eth", **settings)
        )

        self.assertEqual(eth.metadata["label_set"], "eth")
        self.assertEqual(eth.metadata["eth_unavailable_rows_excluded"], 2)
        self.assertEqual(eth.metadata["accepted_rows"], recalib.metadata["accepted_rows"])
        pd.testing.assert_frame_equal(eth.pairwise_distances, recalib.pairwise_distances)
        pd.testing.assert_frame_equal(eth.user_profiles, recalib.user_profiles)
        sources = eth.feature_catalog.set_index("family")["source_columns"]
        self.assertEqual(sources["gaze_direction"], "eth_label_x,eth_label_y,eth_label_z")
        self.assertEqual(
            sources["gaze_origin"], "eth_face_center_x,eth_face_center_y,eth_face_center_z"
        )

    def test_eth_labels_reject_head_pose_and_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "head_rotation"):
            AnalysisConfig(feature_families=("head_rotation",), label_set="eth")
        with self.assertRaisesRegex(ValueError, "label_set"):
            AnalysisConfig(label_set="other")
        frame = pd.DataFrame([_rich_row("00", "00", "9-point", 1)])
        with self.assertRaisesRegex(ValueError, "eth_label_x"):
            analyze_frame(frame, AnalysisConfig(
                feature_families=("gaze_direction",), label_set="eth",
                bootstrap_iterations=0, permutation_iterations=0,
            ))

    def test_command_line_eth_defaults_skip_head_pose(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "summary.csv"
            output_path = Path(directory) / "report"
            _eth_frame().to_csv(input_path, index=False)
            completed = subprocess.run(
                [
                    sys.executable, "-m", "statistical_analysis", str(input_path),
                    "--output", str(output_path), "--labels", "eth", "--views", "raw",
                    "--bootstrap", "0", "--permutations", "0", "--no-progress",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            families = set(pd.read_csv(output_path / "feature_catalog.csv")["family"])
            self.assertNotIn("head_rotation", families)
            self.assertNotIn("head_translation", families)
            self.assertIn("gaze_direction", families)


if __name__ == "__main__":
    unittest.main()
