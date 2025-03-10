# Group Scholar Partner Renewal Tracker

A lightweight CLI that turns partner engagement exports into renewal risk signals. It highlights expiring contracts, stale relationships, and low engagement so the team can prioritize outreach and renewal planning quickly.

## Features
- Scores renewal risk using contact recency, contract horizon, engagement, issues, and activity signals
- Buckets partners into high/medium/low tiers with explainable reasons
- Outputs a concise summary plus a ranked list of top at-risk partners
- Emits JSON reports for downstream dashboards
- Works with simple CSV exports (aliases for common header names)

## Usage

```bash
python3 partner_renewal_tracker.py --input sample/partners.csv --as-of 2026-02-07 --json-out report.json
```

### Options
- `--top` (default 10): how many partners to list
- `--stale-contact-days` (default 45)
- `--renewal-window-days` (default 90)
- `--low-engagement-threshold` (default 55)
- `--high-issues-threshold` (default 3)

## Input schema
Expected headers (aliases accepted):
- `partner_id`
- `partner_name`
- `last_contact_date` (YYYY-MM-DD)
- `contract_end_date` (YYYY-MM-DD)
- `engagement_score` (0-100)
- `meetings_last_90`
- `referrals_last_90`
- `issues_open`
- `funding_commitment`

## Output
- Console summary with risk mix, expiring counts, and top at-risk list
- Optional JSON report with full partner risk details

## Tech
- Python 3 (standard library only)

## Next steps
- Add configurable weighting profiles (renewal-heavy vs engagement-heavy)
- Export CSV snapshots for BI pipelines
