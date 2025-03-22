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
