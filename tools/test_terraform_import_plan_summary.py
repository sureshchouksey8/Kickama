#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from terraform_import import (
    ResourceToImport,
    TerraformImporter,
    load_resources_from_csv,
    redact_secret_value,
)


class TerraformImportPlanSummaryTest(unittest.TestCase):
    def test_summary_is_deterministic_and_marks_existing_resources(self):
        importer = TerraformImporter()
        resources = [
            ResourceToImport("aws_s3_bucket", "logs", "bucket-prod-logs"),
            ResourceToImport("aws_instance", "web", "i-0123456789abcdef0"),
        ]

        summary = importer.build_import_plan_summary(
            resources,
            state_resources=["aws_instance.web"],
        )

        self.assertEqual([item["address"] for item in summary], [
            "aws_instance.web",
            "aws_s3_bucket.logs",
        ])
        self.assertTrue(summary[0]["already_imported"])
        self.assertFalse(summary[1]["already_imported"])
        self.assertEqual(summary[0]["provider_type"], "aws_instance")
        self.assertEqual(summary[0]["import_id"], "i-0123456789abcdef0")

    def test_explicit_terraform_address_is_used_for_modules(self):
        importer = TerraformImporter()
        resources = [
            ResourceToImport(
                "aws_security_group",
                "web",
                "sg-12345678",
                terraform_address="module.network.aws_security_group.web",
            )
        ]

        summary = importer.build_import_plan_summary(resources, state_resources=[])

        self.assertEqual(summary[0]["address"], "module.network.aws_security_group.web")
        self.assertEqual(summary[0]["provider_type"], "aws_security_group")

    def test_secret_looking_import_ids_are_redacted(self):
        self.assertEqual(redact_secret_value("client_secret=s3cr3t-value"), "client_secret=<redacted>")
        self.assertEqual(redact_secret_value("AKIAIOSFODNN7EXAMPLE"), "AKIA...MPLE<redacted>")
        self.assertEqual(redact_secret_value("plain-bucket-name"), "plain-bucket-name")

    def test_write_json_summary_from_csv_without_terraform(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "imports.csv"
            out_path = Path(tmpdir) / "summary.json"
            csv_path.write_text(
                "type,name,id,address,state_file\n"
                "aws_s3_bucket,logs,bucket-prod-logs,,custom.tfstate\n"
                "aws_iam_role,admin,AKIAIOSFODNN7EXAMPLE,module.iam.aws_iam_role.admin,custom.tfstate\n",
                encoding="utf-8",
            )

            resources = load_resources_from_csv(str(csv_path))
            rendered = TerraformImporter().write_import_plan_summary(
                resources,
                str(out_path),
                output_format="json",
                state_resources=[],
            )

            self.assertEqual(out_path.read_text(encoding="utf-8"), rendered)
            data = json.loads(rendered)
            self.assertEqual([item["address"] for item in data], [
                "aws_s3_bucket.logs",
                "module.iam.aws_iam_role.admin",
            ])
            self.assertEqual(data[1]["import_id"], "AKIA...MPLE<redacted>")


if __name__ == "__main__":
    unittest.main()
