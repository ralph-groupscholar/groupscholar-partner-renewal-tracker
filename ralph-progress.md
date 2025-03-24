# Ralph Progress Log

## 2026-02-07
- Started Group Scholar Partner Renewal Tracker, a Python CLI that scores partner renewal risk from CSVs.
- Implemented risk scoring, tiering, console summary, and JSON reporting.
- Added sample dataset and documentation.
- Added funding-at-risk summaries, value-at-risk ranking, and warnings for missing commitments.
- Expanded console/JSON outputs and documented the new controls.
- Added contract horizon/funding buckets plus an action queue with prioritized renewal/outreach recommendations and JSON output.
- Added an owner risk snapshot with expiring/value-at-risk summaries plus JSON output and CLI control for top owners.

## 2026-02-08
- Added configurable weight profiles/overrides and Postgres export support for renewal runs.
- Created production schema/tables and seeded the database with sample partner data.
- Documented Postgres usage and added psycopg requirements.

## 2026-02-08
- Added CSV export outputs for partner risk, action queue, and owner summaries.

## 2026-02-08
- Added renewal calendar buckets (overdue, upcoming months, missing contracts) with console, JSON, HTML, and CSV export support.
- Expanded HTML report to include renewal calendar table and added coverage for calendar bucketing.
- Updated README with calendar/HTML usage and options.

## 2026-02-08
- Added an SVG renewal funding chart to the HTML report with legend and labels.
- Extended HTML report tests to validate the chart output.
- Updated README to reflect the chart and new next step.

## 2026-02-08
- Repaired Postgres export formatting for owner summary and renewal calendar inserts.
