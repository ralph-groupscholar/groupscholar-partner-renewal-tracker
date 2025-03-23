# Group Scholar Partner Renewal Tracker

A lightweight CLI that turns partner engagement exports into renewal risk signals. It highlights expiring contracts, stale relationships, and low engagement so the team can prioritize outreach and renewal planning quickly.

## Features
- Scores renewal risk using contact recency, contract horizon, engagement, issues, and activity signals
- Buckets partners into high/medium/low tiers with explainable reasons
- Outputs a concise summary plus ranked lists for risk and renewal value at risk
- Highlights funding commitments tied to expiring or stale relationships
- Adds an action queue with recommended next steps and priority scores
- Provides an owner risk snapshot to focus renewal outreach by owner
- Emits JSON and HTML reports for downstream dashboards
- Exports CSV snapshots for partners, actions, and owner summaries
- Adds a renewal calendar (overdue + upcoming months) with optional CSV export and SVG chart
- Works with simple CSV exports (aliases for common header names)

## Usage

```bash
python3 partner_renewal_tracker.py --input sample/partners.csv --as-of 2026-02-07 --json-out report.json
```

Generate an HTML report and renewal calendar export:

```bash
python3 partner_renewal_tracker.py --input sample/partners.csv --html-out report.html --calendar-csv-out calendar.csv
```

Export CSV snapshots:

```bash
python3 partner_renewal_tracker.py --input sample/partners.csv --csv-out partners.csv --actions-csv-out actions.csv --owners-csv-out owners.csv --calendar-csv-out calendar.csv
```

Store a run in Postgres (production):

```bash
export GS_PG_DSN="postgresql://USER:PASSWORD@HOST:PORT/DATABASE"
python3 partner_renewal_tracker.py --input sample/partners.csv --export-postgres --run-label "baseline"
```

Note: Postgres export is intended for deployed/production usage. Do not point it at local dev databases.

### Options
- `--top` (default 10): how many partners to list
- `--top-value` (default 5): how many value-at-risk partners to list
- `--top-actions` (default 8): how many action queue entries to list
- `--top-owners` (default 6): how many owners to list in snapshot
- `--stale-contact-days` (default 45)
- `--renewal-window-days` (default 90)
- `--low-engagement-threshold` (default 55)
- `--high-issues-threshold` (default 3)
- `--weight-profile` (balanced, engagement-heavy, renewal-heavy)
- `--weight-contact`, `--weight-contract`, `--weight-engagement`, `--weight-issues`, `--weight-meetings`, `--weight-referrals`
- `--csv-out`: CSV export path for partner risk snapshot
- `--actions-csv-out`: CSV export path for action queue snapshot
- `--owners-csv-out`: CSV export path for owner summary snapshot
- `--calendar-csv-out`: CSV export path for renewal calendar snapshot
- `--calendar-months` (default 6): months to include in renewal calendar
- `--export-postgres`: store results in Postgres (requires env vars)
- `--run-label`: optional label stored with the Postgres run

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
- Console summary with risk mix, expiring counts, and funding-at-risk totals
- Action queue with renewal/outreach recommendations and priority scoring
- Optional JSON report with full partner risk details and value-at-risk calculations

## Tech
- Python 3
- psycopg (for Postgres export)

## Next steps
- Add per-owner renewal calendar breakdowns
