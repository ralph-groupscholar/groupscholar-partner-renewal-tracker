#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

DATE_FMT = "%Y-%m-%d"

ALIASES = {
    "partner_id": ["partner_id", "partnerid", "id"],
    "partner_name": ["partner_name", "name", "partner"],
    "last_contact_date": ["last_contact_date", "last_contact", "last_contacted"],
    "contract_end_date": ["contract_end_date", "contract_end", "renewal_date"],
    "engagement_score": ["engagement_score", "engagement", "health_score"],
    "meetings_last_90": ["meetings_last_90", "meetings_90", "meetings_last_3mo"],
    "referrals_last_90": ["referrals_last_90", "referrals_90", "referrals_last_3mo"],
    "issues_open": ["issues_open", "open_issues", "issues"],
    "funding_commitment": ["funding_commitment", "commitment", "annual_commitment"],
}


@dataclass
class PartnerRecord:
    partner_id: str
    partner_name: str
    last_contact_date: Optional[date]
    contract_end_date: Optional[date]
    engagement_score: float
    meetings_last_90: int
    referrals_last_90: int
    issues_open: int
    funding_commitment: float


@dataclass
class PartnerRisk:
    record: PartnerRecord
    days_since_contact: Optional[int]
    days_to_contract_end: Optional[int]
    risk_score: int
    risk_tier: str
    expired: bool
    reasons: List[str]


def parse_date(value: str) -> Optional[date]:
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, DATE_FMT).date()
    except ValueError:
        return None


def parse_int(value: str) -> int:
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return 0


def parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return 0.0


def pick_field(row: Dict[str, str], field: str) -> str:
    for alias in ALIASES[field]:
        if alias in row and row[alias] is not None:
            return row[alias]
    return ""


def compute_risk(
    record: PartnerRecord,
    as_of: date,
    stale_contact_days: int,
    renewal_window_days: int,
    low_engagement_threshold: float,
    high_issues_threshold: int,
) -> PartnerRisk:
    reasons: List[str] = []
    score = 0

    days_since_contact = None
    if record.last_contact_date:
        days_since_contact = (as_of - record.last_contact_date).days
        if days_since_contact >= stale_contact_days:
            score += 25
            reasons.append("stale_contact")
        elif days_since_contact >= stale_contact_days // 2:
            score += 10
            reasons.append("contact_cooling")
    else:
        score += 20
        reasons.append("missing_contact_date")

    days_to_contract_end = None
    expired = False
    if record.contract_end_date:
        days_to_contract_end = (record.contract_end_date - as_of).days
        if days_to_contract_end < 0:
            score += 20
            expired = True
            reasons.append("contract_expired")
        elif days_to_contract_end <= renewal_window_days:
            score += 30
            reasons.append("renewal_window")
        elif days_to_contract_end <= renewal_window_days * 2:
            score += 10
            reasons.append("renewal_horizon")
    else:
        score += 15
        reasons.append("missing_contract_end")

    if record.engagement_score < low_engagement_threshold:
        score += 20
        reasons.append("low_engagement")
    elif record.engagement_score < low_engagement_threshold + 10:
        score += 10
        reasons.append("soft_engagement")

    if record.issues_open >= high_issues_threshold:
        score += 15
        reasons.append("issues_high")
    elif record.issues_open > 0:
        score += 5
        reasons.append("issues_open")

    if record.meetings_last_90 == 0:
        score += 10
        reasons.append("no_recent_meetings")

    if record.referrals_last_90 == 0:
        score += 5
        reasons.append("no_recent_referrals")

    score = min(score, 100)

    if score >= 70:
        tier = "high"
    elif score >= 40:
        tier = "medium"
    else:
        tier = "low"

    return PartnerRisk(
        record=record,
        days_since_contact=days_since_contact,
        days_to_contract_end=days_to_contract_end,
        risk_score=score,
        risk_tier=tier,
        expired=expired,
        reasons=reasons,
    )


def load_partners(path: str) -> Tuple[List[PartnerRecord], List[str]]:
    warnings: List[str] = []
    records: List[PartnerRecord] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing headers.")
        for idx, row in enumerate(reader, start=2):
            partner_name = pick_field(row, "partner_name").strip()
            partner_id = pick_field(row, "partner_id").strip()
            funding_raw = pick_field(row, "funding_commitment").strip()
            if not partner_name:
                warnings.append(f"Row {idx}: missing partner_name")
            if not partner_id:
                partner_id = f"row-{idx}"
            if not funding_raw:
                warnings.append(f"Row {idx}: missing funding_commitment")
            record = PartnerRecord(
                partner_id=partner_id,
                partner_name=partner_name or "Unknown",
                last_contact_date=parse_date(pick_field(row, "last_contact_date")),
                contract_end_date=parse_date(pick_field(row, "contract_end_date")),
                engagement_score=parse_float(pick_field(row, "engagement_score")),
                meetings_last_90=parse_int(pick_field(row, "meetings_last_90")),
                referrals_last_90=parse_int(pick_field(row, "referrals_last_90")),
                issues_open=parse_int(pick_field(row, "issues_open")),
                funding_commitment=parse_float(funding_raw),
            )
            records.append(record)
    return records, warnings


def compute_value_risk(risk: PartnerRisk) -> float:
    return round(risk.record.funding_commitment * (risk.risk_score / 100.0), 2)


def summarize(risks: List[PartnerRisk], renewal_window_days: int, stale_contact_days: int) -> Dict[str, float]:
    total = len(risks)
    high = sum(1 for r in risks if r.risk_tier == "high")
    medium = sum(1 for r in risks if r.risk_tier == "medium")
    low = sum(1 for r in risks if r.risk_tier == "low")
    expiring = sum(1 for r in risks if r.days_to_contract_end is not None and r.days_to_contract_end <= renewal_window_days)
    stale = sum(1 for r in risks if r.days_since_contact is not None and r.days_since_contact >= stale_contact_days)
    total_funding = sum(r.record.funding_commitment for r in risks)
    high_risk_funding = sum(r.record.funding_commitment for r in risks if r.risk_tier == "high")
    expiring_funding = sum(
        r.record.funding_commitment
        for r in risks
        if r.days_to_contract_end is not None and r.days_to_contract_end <= renewal_window_days
    )
    stale_funding = sum(
        r.record.funding_commitment
        for r in risks
        if r.days_since_contact is not None and r.days_since_contact >= stale_contact_days
    )
    avg_value_risk = (
        sum(compute_value_risk(r) for r in risks) / total
        if total else 0.0
    )
    avg_engagement = (
        sum(r.record.engagement_score for r in risks) / total
        if total else 0.0
    )
    avg_score = (
        sum(r.risk_score for r in risks) / total
        if total else 0.0
    )
    return {
        "total_partners": total,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "expiring_within_window": expiring,
        "stale_contacts": stale,
        "average_engagement": round(avg_engagement, 2),
        "average_risk_score": round(avg_score, 2),
        "total_funding_commitment": round(total_funding, 2),
        "high_risk_funding": round(high_risk_funding, 2),
        "expiring_funding_within_window": round(expiring_funding, 2),
        "stale_contact_funding": round(stale_funding, 2),
        "average_value_at_risk": round(avg_value_risk, 2),
    }


def render_console(summary: Dict[str, float], top: List[PartnerRisk], top_value: List[PartnerRisk]) -> None:
    print("Partner Renewal Tracker")
    print("=" * 26)
    print(f"Total partners: {summary['total_partners']}")
    print(f"Risk mix: high {summary['high_risk']} | medium {summary['medium_risk']} | low {summary['low_risk']}")
    print(f"Expiring within window: {summary['expiring_within_window']}")
    print(f"Stale contacts: {summary['stale_contacts']}")
    print(f"Average engagement: {summary['average_engagement']}")
    print(f"Average risk score: {summary['average_risk_score']}")
    print(f"Total funding commitment: {summary['total_funding_commitment']}")
    print(f"High-risk funding: {summary['high_risk_funding']}")
    print(f"Expiring funding within window: {summary['expiring_funding_within_window']}")
    print(f"Stale-contact funding: {summary['stale_contact_funding']}")
    print(f"Average value at risk: {summary['average_value_at_risk']}")
    print("\nTop at-risk partners")
    print("-" * 26)
    if not top:
        print("No partners found.")
        return
    for risk in top:
        record = risk.record
        reasons = ", ".join(risk.reasons) if risk.reasons else "none"
        days_to_end = "n/a" if risk.days_to_contract_end is None else str(risk.days_to_contract_end)
        days_since = "n/a" if risk.days_since_contact is None else str(risk.days_since_contact)
        print(
            f"{record.partner_name} ({record.partner_id}) | score {risk.risk_score} | tier {risk.risk_tier} | "
            f"days_to_end {days_to_end} | days_since_contact {days_since} | reasons: {reasons}"
        )
    if top_value:
        print("\nTop value at risk")
        print("-" * 26)
        for risk in top_value:
            record = risk.record
            value_risk = compute_value_risk(risk)
            print(
                f"{record.partner_name} ({record.partner_id}) | commitment {record.funding_commitment} | "
                f"value_at_risk {value_risk} | score {risk.risk_score} | tier {risk.risk_tier}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Track partner renewal risk signals from CSV exports.")
    parser.add_argument("--input", required=True, help="Path to partner CSV export.")
    parser.add_argument("--as-of", help="Override as-of date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--json-out", help="Optional path for JSON report output.")
    parser.add_argument("--top", type=int, default=10, help="How many top at-risk partners to show.")
    parser.add_argument("--top-value", type=int, default=5, help="How many top value-at-risk partners to show.")
    parser.add_argument("--stale-contact-days", type=int, default=45)
    parser.add_argument("--renewal-window-days", type=int, default=90)
    parser.add_argument("--low-engagement-threshold", type=float, default=55.0)
    parser.add_argument("--high-issues-threshold", type=int, default=3)

    args = parser.parse_args()

    as_of = date.today()
    if args.as_of:
        parsed = parse_date(args.as_of)
        if not parsed:
            raise SystemExit("Invalid --as-of date. Use YYYY-MM-DD.")
        as_of = parsed

    records, warnings = load_partners(args.input)
    risks = [
        compute_risk(
            record,
            as_of,
            args.stale_contact_days,
            args.renewal_window_days,
            args.low_engagement_threshold,
            args.high_issues_threshold,
        )
        for record in records
    ]

    risks.sort(key=lambda r: (r.risk_score, r.record.partner_name), reverse=True)
    value_ranked = sorted(
        risks,
        key=lambda r: (compute_value_risk(r), r.risk_score, r.record.partner_name),
        reverse=True,
    )
    summary = summarize(risks, args.renewal_window_days, args.stale_contact_days)

    render_console(summary, risks[: args.top], value_ranked[: args.top_value])

    if warnings:
        print("\nWarnings")
        print("-" * 26)
        for warning in warnings:
            print(warning)

    if args.json_out:
        payload = {
            "as_of": as_of.strftime(DATE_FMT),
            "summary": summary,
            "partners": [
                {
                    "partner_id": risk.record.partner_id,
                    "partner_name": risk.record.partner_name,
                    "last_contact_date": risk.record.last_contact_date.strftime(DATE_FMT)
                    if risk.record.last_contact_date
                    else None,
                    "contract_end_date": risk.record.contract_end_date.strftime(DATE_FMT)
                    if risk.record.contract_end_date
                    else None,
                    "engagement_score": risk.record.engagement_score,
                    "meetings_last_90": risk.record.meetings_last_90,
                    "referrals_last_90": risk.record.referrals_last_90,
                    "issues_open": risk.record.issues_open,
                    "funding_commitment": risk.record.funding_commitment,
                    "value_at_risk": compute_value_risk(risk),
                    "days_since_contact": risk.days_since_contact,
                    "days_to_contract_end": risk.days_to_contract_end,
                    "risk_score": risk.risk_score,
                    "risk_tier": risk.risk_tier,
                    "expired": risk.expired,
                    "reasons": risk.reasons,
                }
                for risk in risks
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
