import unittest
from pathlib import Path
from datetime import date

import partner_renewal_tracker as prt


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.sample_path = Path(__file__).resolve().parents[1] / "sample" / "partners.csv"
        self.as_of = date(2026, 2, 7)

    def test_weight_overrides_clamped(self):
        weights = prt.resolve_weights(
            "balanced",
            contact=-1.0,
            contract=1.5,
            engagement=None,
            issues=None,
            meetings=0.0,
            referrals=2.0,
        )
        self.assertEqual(weights["contact"], 0.0)
        self.assertEqual(weights["contract"], 1.5)
        self.assertEqual(weights["meetings"], 0.0)
        self.assertEqual(weights["referrals"], 2.0)

    def test_html_report_contains_partner(self):
        records, warnings = prt.load_partners(str(self.sample_path))
        weights = prt.resolve_weights("balanced", None, None, None, None, None, None)
        risks = [
            prt.compute_risk(
                record,
                self.as_of,
                stale_contact_days=45,
                renewal_window_days=90,
                low_engagement_threshold=55.0,
                high_issues_threshold=3,
                weights=weights,
            )
            for record in records
        ]
        risks.sort(key=lambda r: r.risk_score, reverse=True)
        actions = [
            prt.build_action_plan(
                risk,
                renewal_window_days=90,
                stale_contact_days=45,
                low_engagement_threshold=55.0,
                high_issues_threshold=3,
            )
            for risk in risks
        ]
        summary = prt.summarize(risks, 90, 45, weights)
        owners = prt.build_owner_summary(risks, 90, 45)
        calendar = prt.build_renewal_calendar(risks, self.as_of, 6)
        owner_calendars = prt.build_owner_renewal_calendars(risks, self.as_of, 6)
        reasons = prt.build_reason_summary(risks)

        output_path = Path(__file__).resolve().parent / "report.html"
        prt.write_html_report(
            str(output_path),
            self.as_of,
            summary,
            risks[:3],
            risks[:2],
            actions[:4],
            owners[:2],
            owner_calendars[:2],
            calendar,
            reasons,
            warnings,
        )
        html = output_path.read_text(encoding="utf-8")
        self.assertIn("Partner Renewal Tracker", html)
        self.assertIn("Top at-risk partners", html)
        self.assertIn("Blue Oak Philanthropy", html)
        self.assertIn("Owner renewal calendars", html)
        self.assertIn("Risk drivers", html)
        self.assertIn("<svg", html)
        self.assertIn("Renewal funding by bucket", html)
        output_path.unlink(missing_ok=True)

    def test_renewal_calendar_buckets(self):
        records, _warnings = prt.load_partners(str(self.sample_path))
        weights = prt.resolve_weights("balanced", None, None, None, None, None, None)
        risks = [
            prt.compute_risk(
                record,
                self.as_of,
                stale_contact_days=45,
                renewal_window_days=90,
                low_engagement_threshold=55.0,
                high_issues_threshold=3,
                weights=weights,
            )
            for record in records
        ]
        calendar = prt.build_renewal_calendar(risks, self.as_of, 6)
        bucket_map = {item["bucket"]: item for item in calendar}
        self.assertEqual(bucket_map["overdue"]["expiring_partners"], 2)
        self.assertEqual(bucket_map["2026-02"]["expiring_partners"], 2)
        self.assertEqual(bucket_map["2026-03"]["expiring_partners"], 2)
        self.assertEqual(bucket_map["2026-04"]["expiring_partners"], 1)
        self.assertEqual(bucket_map["2026-05"]["expiring_partners"], 1)
        self.assertEqual(bucket_map["2026-06"]["expiring_partners"], 1)
        self.assertEqual(bucket_map["missing_contract"]["expiring_partners"], 0)

    def test_owner_renewal_calendar(self):
        records, _warnings = prt.load_partners(str(self.sample_path))
        weights = prt.resolve_weights("balanced", None, None, None, None, None, None)
        risks = [
            prt.compute_risk(
                record,
                self.as_of,
                stale_contact_days=45,
                renewal_window_days=90,
                low_engagement_threshold=55.0,
                high_issues_threshold=3,
                weights=weights,
            )
            for record in records
        ]
        owner_calendars = prt.build_owner_renewal_calendars(risks, self.as_of, 6)
        calendar_map = {item["owner"]: item for item in owner_calendars}
        ava_calendar = {item["bucket"]: item for item in calendar_map["Ava Chen"]["calendar"]}
        jordan_calendar = {item["bucket"]: item for item in calendar_map["Jordan Lee"]["calendar"]}
        priya_calendar = {item["bucket"]: item for item in calendar_map["Priya Nair"]["calendar"]}

        self.assertEqual(ava_calendar["2026-02"]["expiring_partners"], 1)
        self.assertEqual(ava_calendar["2026-03"]["expiring_partners"], 2)
        self.assertEqual(ava_calendar["2026-05"]["expiring_partners"], 1)
        self.assertEqual(jordan_calendar["overdue"]["expiring_partners"], 1)
        self.assertEqual(jordan_calendar["2026-02"]["expiring_partners"], 1)
        self.assertEqual(jordan_calendar["2026-06"]["expiring_partners"], 1)
        self.assertEqual(priya_calendar["overdue"]["expiring_partners"], 1)
        self.assertEqual(priya_calendar["2026-04"]["expiring_partners"], 1)
        self.assertEqual(priya_calendar["2026-07"]["expiring_partners"], 1)


if __name__ == "__main__":
    unittest.main()
