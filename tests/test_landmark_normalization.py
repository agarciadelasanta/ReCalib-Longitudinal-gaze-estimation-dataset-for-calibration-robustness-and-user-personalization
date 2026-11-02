import unittest

import numpy as np
import pandas as pd

from statistical_analysis.__main__ import (
    _config_from_args, build_parser, build_compare_parser,
)
from statistical_analysis.core import (
    AnalysisConfig, FEATURE_COLUMNS, LANDMARK_FAMILIES, LANDMARK_IDS,
    _prepare_accepted_frame, analyze_frame,
)
from statistical_analysis.comparison import DEFAULT_COMPARISON_FAMILIES, compare_frames


def samples():
    rows = []
    for user in range(2):
        for session in range(2):
            row = dict(user_id=str(user), session_id=str(session),
                       task_type="9-point", discarded=False,
                       img_res_width=640, img_res_height=480)
            for i, landmark in enumerate(LANDMARK_IDS):
                row[f"lm_{landmark}_x"] = 120 + (i % 6) * 8 + user + session
                row[f"lm_{landmark}_y"] = 80 + (i // 6) * 8 + user + session
            rows.append(row)
    return pd.DataFrame(rows)


class LandmarkNormalizationTests(unittest.TestCase):
    def test_cli_and_library_defaults_exclude_landmarks_and_cli_can_opt_in(self):
        self.assertTrue(set(LANDMARK_FAMILIES).isdisjoint(AnalysisConfig().feature_families))
        self.assertTrue(set(LANDMARK_FAMILIES).isdisjoint(DEFAULT_COMPARISON_FAMILIES))
        for parser, inputs in ((build_parser(), ["a.csv"]),
                               (build_compare_parser(), ["a.csv", "b.csv"])):
            args = [*inputs, "--output", "output"]
            default = _config_from_args(parser.parse_args(args), parser)
            self.assertTrue(set(LANDMARK_FAMILIES).isdisjoint(default.feature_families))
            enabled = _config_from_args(parser.parse_args([*args, "--include-landmarks"]), parser)
            self.assertTrue(set(LANDMARK_FAMILIES).issubset(enabled.feature_families))

    def test_per_sample_resolution_changes_leave_both_analyses_unchanged(self):
        original = samples()
        resized = original.copy()
        # Different horizontal/vertical resizing per row exercises both dimensions.
        for axis, dimension, factors in (
            ("x", "img_res_width", [1, 2, 4, 2]),
            ("y", "img_res_height", [2, 4, 1, 2]),
        ):
            for column in [dimension, *(f"lm_{i}_{axis}" for i in LANDMARK_IDS)]:
                resized[column] *= np.array(factors)
        config = AnalysisConfig(feature_families=LANDMARK_FAMILIES, views=("raw",),
                                bootstrap_iterations=0, permutation_iterations=0)
        prepared = _prepare_accepted_frame(original, config)
        normalized = _prepare_accepted_frame(resized, config)
        for family in LANDMARK_FAMILIES:
            pd.testing.assert_frame_equal(prepared[list(FEATURE_COLUMNS[family])],
                                          normalized[list(FEATURE_COLUMNS[family])])
        expected_x = original[[f"lm_{i}_x" for i in LANDMARK_IDS]].mean(axis=1) / 640
        np.testing.assert_allclose(prepared._landmark_center_x, expected_x)
        pd.testing.assert_frame_equal(analyze_frame(original, config).pairwise_distances,
                                      analyze_frame(resized, config).pairwise_distances)
        pd.testing.assert_frame_equal(compare_frames(original, original, config).dataset_distances,
                                      compare_frames(original, resized, config).dataset_distances)

    def test_landmarks_require_valid_image_dimensions(self):
        config = AnalysisConfig(feature_families=LANDMARK_FAMILIES, views=("raw",))
        frame = samples()
        with self.assertRaisesRegex(ValueError, "img_res_width"):
            _prepare_accepted_frame(frame.drop(columns="img_res_width"), config)
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite, positive"):
                _prepare_accepted_frame(frame.assign(img_res_height=value), config)
