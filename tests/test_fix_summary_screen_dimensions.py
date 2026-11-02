from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from statistical_analysis.fix_summary_screen_dimensions import inspect_csv, main


class FixSummaryScreenDimensionsTests(unittest.TestCase):
    def _write(self, directory: Path, rows: list[dict[str, object]]) -> Path:
        path = directory / "dataset_summary_example.csv"
        fieldnames = [
            "user_id",
            "pog_px_x",
            "pog_px_y",
            "screen_width_px",
            "screen_height_px",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_dry_run_reports_changes_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary),
                [{
                    "user_id": "00",
                    "pog_px_x": 2654,
                    "pog_px_y": 1742,
                    "screen_width_px": 1368,
                    "screen_height_px": 912,
                }],
            )
            before = path.read_bytes()

            self.assertEqual(main([str(path)]), 0)

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(inspect_csv(path).changed_rows, 1)

    def test_apply_updates_dimensions_and_creates_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary),
                [{
                    "user_id": "00",
                    "pog_px_x": 2726,
                    "pog_px_y": 1814,
                    "screen_width_px": "",
                    "screen_height_px": "",
                }],
            )
            original = path.read_bytes()

            self.assertEqual(main([str(path), "--apply"]), 0)

            backup = Path(f"{path}.bak")
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_bytes(), original)
            with path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["screen_width_px"], "2736")
            self.assertEqual(row["screen_height_px"], "1824")
            self.assertEqual(inspect_csv(path).changed_rows, 0)

    def test_out_of_range_pog_is_rejected_before_modification(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary),
                [{
                    "user_id": "00",
                    "pog_px_x": 3000,
                    "pog_px_y": 100,
                    "screen_width_px": 1368,
                    "screen_height_px": 912,
                }],
            )
            before = path.read_bytes()

            with self.assertRaises(SystemExit):
                main([str(path), "--apply"])

            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(Path(f"{path}.bak").exists())


if __name__ == "__main__":
    unittest.main()
