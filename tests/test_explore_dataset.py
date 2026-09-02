import argparse
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "visualization" / "explore_dataset.py"
SPEC = importlib.util.spec_from_file_location("explore_dataset", MODULE_PATH)
explore_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(explore_dataset)


class ExploreDatasetTest(unittest.TestCase):
    def setUp(self):
        self.screen = explore_dataset.ScreenGeometry(
            width_cm=26.0,
            height_cm=17.3,
            width_px=2736.0,
            height_px=1824.0,
        )
        self.frame = pd.DataFrame(
            {
                "user_id": [7, 7, 7],
                "session_id": [1, 1, 1],
                "task_id": [0, 1, 1],
                "task_type": ["9-point", "16-point", "16-point"],
                "head_rot_x": [0.0, math.pi / 6, math.pi / 4],
                "head_rot_y": [math.pi / 2, 0.0, -math.pi / 6],
                "head_rot_z": [0.0, -math.pi / 4, math.pi / 6],
                "head_trans_x": [-20.0, 10.0, 30.0],
                "head_trans_y": [40.0, 50.0, 60.0],
                "head_trans_z": [300.0, 320.0, 340.0],
                "pog_px_x": [1368, 10, 10],
                "pog_px_y": [912, 10, 10],
            }
        )

    def test_setup_config_applies_zoom_and_converts_mm_to_cm(self):
        parser = argparse.ArgumentParser()
        screen = explore_dataset.read_screen_geometry(
            REPO_ROOT / "docs" / "setup_config.json", parser
        )

        self.assertEqual(screen.width_px, 2736)
        self.assertEqual(screen.height_px, 1824)
        self.assertEqual(screen.width_cm, 26)
        self.assertEqual(screen.height_cm, 17.3)

    def test_records_support_exact_target_filtering_and_degree_plots(self):
        groups = explore_dataset.make_group_records(self.frame, self.screen)
        points = explore_dataset.make_point_records(self.frame, 100, self.screen)

        center = next(row for row in groups if row["target_key"] == "1368|912")
        repeated = next(row for row in groups if row["target_key"] == "10|10")
        self.assertAlmostEqual(center["target_cm_x"], 13.0)
        self.assertAlmostEqual(center["target_cm_y"], 8.65)
        self.assertEqual(repeated["count"], 2)
        self.assertAlmostEqual(points[1]["head_pitch"], 30.0)
        self.assertAlmostEqual(points[1]["head_roll"], -45.0)
        self.assertAlmostEqual(points[1]["head_translation_x"], 1.0)
        self.assertAlmostEqual(points[1]["head_translation_z"], 32.0)

    def test_report_has_target_cross_filter_and_no_quality_outcome_ui(self):
        args = SimpleNamespace(
            csv=Path("fixture.csv"),
            id_user=None,
            id_session=None,
            id_task=None,
            max_points=100,
        )

        report = explore_dataset.build_report(self.frame, args, self.screen)

        self.assertIn("Screen target cross-filter", report)
        self.assertIn('targetElement.on("plotly_click"', report)
        self.assertIn("Screen x (cm)", report)
        self.assertIn("Pixel position", report)
        self.assertIn("Yaw and pitch coverage by task", report)
        self.assertIn("Population regions", report)
        self.assertIn('type: "histogram2dcontour"', report)
        self.assertIn('<option value="roll-yaw">Roll and yaw</option>', report)
        self.assertIn(
            '<option value="horizontal-vertical">Horizontal and vertical</option>',
            report,
        )
        self.assertIn("function setCoverageAxes(kind, axes)", report)
        self.assertIn("Horizontal and depth translation coverage by task", report)
        self.assertIn("Head translation distributions by task", report)
        self.assertIn("Central 90% head-translation span", report)
        self.assertIn('role="tablist"', report)
        self.assertIn('aria-controls="overview-panel"', report)
        self.assertIn('aria-controls="distributions-panel"', report)
        self.assertIn("function activateTab(button)", report)
        self.assertIn("function initializeCollapsiblePanels()", report)
        self.assertIn('toggle.textContent = expanded ? "Minimize" : "Open"', report)
        self.assertIn('content.querySelectorAll(".js-plotly-plot")', report)
        overview = report[
            report.index('id="overview-panel"') : report.index(
                'id="distributions-panel"'
            )
        ]
        distributions = report[report.index('id="distributions-panel"') :]
        self.assertEqual(distributions.count('<article class="panel wide">'), 7)
        for heading in [
            "Task composition",
            "Longitudinal session coverage",
            "Session breakdown",
        ]:
            self.assertIn(heading, overview)
            self.assertNotIn(heading, distributions)
        for heading in [
            "Screen target cross-filter",
            "Head rotation distributions by task",
            "Central 90% head-pose span",
        ]:
            self.assertIn(heading, distributions)
            self.assertNotIn(heading, overview)
        self.assertLess(report.index("Task composition"), report.index("Screen target cross-filter"))
        self.assertNotIn("Quality outcomes", report)
        self.assertNotIn("status-filter", report)


if __name__ == "__main__":
    unittest.main()
