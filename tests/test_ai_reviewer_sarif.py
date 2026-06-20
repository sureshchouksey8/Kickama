#!/usr/bin/env python3
"""Fixture tests for ai_reviewer SARIF output."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ai_reviewer import (  # noqa: E402
    AiCodeReviewer,
    FileReviewResult,
    ReviewCategory,
    ReviewFinding,
    ReviewSeverity,
)


def finding(severity: ReviewSeverity, line: int) -> ReviewFinding:
    return ReviewFinding(
        id=f"FIXTURE-{severity.value}-{line}",
        severity=severity,
        category=ReviewCategory.SECURITY,
        message=f"{severity.value} fixture finding",
        file_path="fixtures/example.py",
        line_number=line,
        suggestion=f"Resolve {severity.value}",
        rules=[f"FIXTURE-{severity.value.upper()}"],
    )


class SarifOutputTest(unittest.TestCase):
    def test_sarif_contains_one_result_for_each_reviewer_severity(self) -> None:
        severities = [
            ReviewSeverity.CRITICAL,
            ReviewSeverity.HIGH,
            ReviewSeverity.ERROR,
            ReviewSeverity.WARNING,
            ReviewSeverity.INFO,
            ReviewSeverity.SUGGESTION,
        ]
        result = FileReviewResult(
            file_path="fixtures/example.py",
            language="python",
            line_count=20,
            findings=[finding(severity, index + 1) for index, severity in enumerate(severities)],
        )

        sarif = json.loads(AiCodeReviewer().generate_report_sarif([result]))
        output_results = sarif["runs"][0]["results"]

        self.assertEqual(sarif["version"], "2.1.0")
        self.assertEqual(len(output_results), len(severities))
        self.assertEqual(
            {item["properties"]["reviewSeverity"] for item in output_results},
            {severity.value for severity in severities},
        )
        self.assertEqual(
            {
                item["properties"]["reviewSeverity"]: item["level"]
                for item in output_results
            },
            {
                "critical": "error",
                "high": "error",
                "error": "error",
                "warning": "warning",
                "info": "note",
                "suggestion": "note",
            },
        )
        for item in output_results:
            location = item["locations"][0]["physicalLocation"]
            self.assertEqual(location["artifactLocation"]["uri"], "fixtures/example.py")
            self.assertGreaterEqual(location["region"]["startLine"], 1)
            self.assertIn("Suggestion:", item["message"]["text"])


if __name__ == "__main__":
    unittest.main()
