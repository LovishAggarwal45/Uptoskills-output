"""
Unit tests for AlertManager and ReportGenerator.
"""

import unittest
import tempfile
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts.alert_manager import AlertManager
from src.alerts.report_generator import ReportGenerator

class TestAlertsAndReports(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "alerts.log"
        self.csv_file = Path(self.temp_dir.name) / "report.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_alert_manager(self):
        mgr = AlertManager(confidence_threshold=0.70, log_file=str(self.log_file), enable_console=False)
        events = [
            {"event_id": 1, "action": "Fighting", "confidence": 0.85, "start_time": 2.0, "end_time": 5.0, "duration": 3.0, "start_time_str": "00:00:02", "end_time_str": "00:00:05"},
            {"event_id": 2, "action": "Normal", "confidence": 0.30, "start_time": 6.0, "end_time": 8.0, "duration": 2.0, "start_time_str": "00:00:06", "end_time_str": "00:00:08"}
        ]
        alerts = mgr.process_events("test_vid", events)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["condition"], "Fighting")
        self.assertTrue(self.log_file.exists())

    def test_report_generator(self):
        rep = ReportGenerator(output_csv=str(self.csv_file))
        events = [
            {"event_id": 1, "action": "Fighting", "confidence": 0.88, "start_time": 2.5, "end_time": 6.0, "duration": 3.5, "start_time_str": "00:00:02.50", "end_time_str": "00:00:06.00"}
        ]
        bboxes = {1: (0.2, 0.3, 0.4, 0.5)}
        rep.write_events("vid_01", events, bboxes)

        self.assertTrue(self.csv_file.exists())
        with open(self.csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2) # Header + 1 record
            self.assertEqual(rows[1][0], "vid_01")
            self.assertEqual(rows[1][5], "Fighting")
            self.assertEqual(rows[1][7], "0.2000")

if __name__ == "__main__":
    unittest.main()
