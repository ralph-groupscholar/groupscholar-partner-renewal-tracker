#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

DATE_FMT = "%Y-%m-%d"

ALIASES = {
    "partner_id": ["partner_id", "partnerid", "id"],
    "partner_name": ["partner_name", "name", "partner"],
    "owner": ["owner", "account_owner", "relationship_manager", "partner_owner"],
    "last_contact_date": ["last_contact_date", "last_contact", "last_contacted"],
    "contract_end_date": ["contract_end_date", "contract_end", "renewal_date"],
    "engagement_score": ["engagement_score", "engagement", "health_score"],
    "meetings_last_90": ["meetings_last_90", "meetings_90", "meetings_last_3mo"],
    "referrals_last_90": ["referrals_last_90", "referrals_90", "referrals_last_3mo"],
    "issues_open": ["issues_open", "open_issues", "issues"],
    "funding_commitment": ["funding_commitment", "commitment", "annual_commitment"],
}

WEIGHT_PROFILES = {
    "balanced": {
        "contact": 1.0,
        "contract": 1.0,
        "engagement": 1.0,
        "issues": 1.0,
        "meetings": 1.0,
        "referrals": 1.0,
    },
    "renewal-heavy": {
        "contact": 1.0,
        "contract": 1.35,
        "engagement": 0.95,
        "issues": 1.05,
        "meetings": 0.9,
        "referrals": 0.9,
    },
    "engagement-heavy": {
        "contact": 1.2,
        "contract": 0.9,
        "engagement": 1.25,
        "issues": 1.1,
        "meetings": 1.1,
        "referrals": 1.05,
    },
}

@dataclass
class PartnerRecord:
    partner_id: str
    partner_name: str
    owner: str
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
    action_code: str
    action_note: str
    action_priority: int


@dataclass
class PartnerAction:
    risk: PartnerRisk
    action_score: int
    action: str
    focus: str


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
    weights: Dict[str, float],
) -> PartnerRisk:
    reasons: List[str] = []
    contact_score = 0.0
    contract_score = 0.0
    engagement_score = 0.0
    issues_score = 0.0
    meetings_score = 0.0
    referrals_score = 0.0

    days_since_contact = None
    if record.last_contact_date:
        days_since_contact = (as_of - record.last_contact_date).days
        if days_since_contact >= stale_contact_days:
            contact_score += 25
            reasons.append("stale_contact")
        elif days_since_contact >= stale_contact_days // 2:
            contact_score += 10
            reasons.append("contact_cooling")
    else:
        contact_score += 20
        reasons.append("missing_contact_date")

    days_to_contract_end = None
    expired = False
    if record.contract_end_date:
        days_to_contract_end = (record.contract_end_date - as_of).days
        if days_to_contract_end < 0:
            contract_score += 20
            expired = True
            reasons.append("contract_expired")
        elif days_to_contract_end <= renewal_window_days:
            contract_score += 30
            reasons.append("renewal_window")
        elif days_to_contract_end <= renewal_window_days * 2:
            contract_score += 10
            reasons.append("renewal_horizon")
    else:
        contract_score += 15
        reasons.append("missing_contract_end")

    if record.engagement_score < low_engagement_threshold:
        engagement_score += 20
        reasons.append("low_engagement")
    elif record.engagement_score < low_engagement_threshold + 10:
        engagement_score += 10
        reasons.append("soft_engagement")

    if record.issues_open >= high_issues_threshold:
        issues_score += 15
        reasons.append("issues_high")
    elif record.issues_open > 0:
        issues_score += 5
        reasons.append("issues_open")

    if record.meetings_last_90 == 0:
        meetings_score += 10
        reasons.append("no_recent_meetings")

    if record.referrals_last_90 == 0:
        referrals_score += 5
        reasons.append("no_recent_referrals")

    weighted_score = (
        contact_score * weights["contact"]
        + contract_score * weights["contract"]
        + engagement_score * weights["engagement"]
        + issues_score * weights["issues"]
        + meetings_score * weights["meetings"]
        + referrals_score * weights["referrals"]
    )
    score = min(int(round(weighted_score)), 100)

    if score >= 70:
        tier = "high"
    elif score >= 40:
        tier = "medium"
    else:
        tier = "low"

    action_code, action_note = recommend_action(
        record,
        days_since_contact,
        days_to_contract_end,
        expired,
        stale_contact_days,
        renewal_window_days,
        low_engagement_threshold,
        high_issues_threshold,
    )
    action_priority = compute_action_priority(
        score,
        days_since_contact,
        days_to_contract_end,
        expired,
        stale_contact_days,
        renewal_window_days,
    )

    return PartnerRisk(
        record=record,
        days_since_contact=days_since_contact,
        days_to_contract_end=days_to_contract_end,
        risk_score=score,
        risk_tier=tier,
        expired=expired,
        reasons=reasons,
        action_code=action_code,
        action_note=action_note,
        action_priority=action_priority,
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
            owner = pick_field(row, "owner").strip() or "Unassigned"
            record = PartnerRecord(
                partner_id=partner_id,
                partner_name=partner_name or "Unknown",
                owner=owner,
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


def compute_action_priority(
    risk_score: int,
    days_since_contact: Optional[int],
    days_to_contract_end: Optional[int],
    expired: bool,
    stale_contact_days: int,
    renewal_window_days: int,
) -> int:
    priority = risk_score
    if expired:
        priority += 25
    if days_to_contract_end is not None:
        if days_to_contract_end <= renewal_window_days:
            priority += 20
        elif days_to_contract_end <= renewal_window_days * 2:
            priority += 10
    if days_since_contact is None or days_since_contact >= stale_contact_days:
        priority += 10
    return min(priority, 120)


def recommend_action(
    record: PartnerRecord,
    days_since_contact: Optional[int],
    days_to_contract_end: Optional[int],
    expired: bool,
    stale_contact_days: int,
    renewal_window_days: int,
    low_engagement_threshold: float,
    high_issues_threshold: int,
) -> Tuple[str, str]:
    if expired:
        return "renewal_overdue", "Renewal overdue: re-engage + confirm commitment"
    if days_to_contract_end is not None and days_to_contract_end <= renewal_window_days:
        return "launch_renewal", "Launch renewal: confirm intent + timeline"
    if days_since_contact is None or days_since_contact >= stale_contact_days:
        return "reconnect", "Reconnect: schedule check-in"
    if record.engagement_score < low_engagement_threshold:
        return "rebuild_engagement", "Rebuild engagement: share impact + invite"
    if record.issues_open >= high_issues_threshold or record.issues_open > 0:
        return "resolve_issues", "Resolve issues: close blockers"
    if record.meetings_last_90 == 0:
        return "schedule_meeting", "Schedule meeting: align on goals"
    if record.referrals_last_90 == 0:
        return "spark_referrals", "Spark referrals: propose scholar match"
    return "monitor", "Monitor: steady state"


def summarize(
    risks: List[PartnerRisk],
    renewal_window_days: int,
    stale_contact_days: int,
    weights: Dict[str, float],
) -> Dict[str, object]:
    total = len(risks)
    high = sum(1 for r in risks if r.risk_tier == "high")
    medium = sum(1 for r in risks if r.risk_tier == "medium")
    low = sum(1 for r in risks if r.risk_tier == "low")
    expired = sum(1 for r in risks if r.days_to_contract_end is not None and r.days_to_contract_end < 0)
    expiring = sum(1 for r in risks if r.days_to_contract_end is not None and r.days_to_contract_end <= renewal_window_days)
    upcoming = sum(
        1
        for r in risks
        if r.days_to_contract_end is not None and renewal_window_days < r.days_to_contract_end <= renewal_window_days * 2
    )
    stale = sum(1 for r in risks if r.days_since_contact is not None and r.days_since_contact >= stale_contact_days)
    total_funding = sum(r.record.funding_commitment for r in risks)
    high_risk_funding = sum(r.record.funding_commitment for r in risks if r.risk_tier == "high")
    expired_funding = sum(
        r.record.funding_commitment
        for r in risks
        if r.days_to_contract_end is not None and r.days_to_contract_end < 0
    )
    expiring_funding = sum(
        r.record.funding_commitment
        for r in risks
        if r.days_to_contract_end is not None and r.days_to_contract_end <= renewal_window_days
    )
    upcoming_funding = sum(
        r.record.funding_commitment
        for r in risks
        if r.days_to_contract_end is not None and renewal_window_days < r.days_to_contract_end <= renewal_window_days * 2
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
        "weight_profile": weights,
        "total_partners": total,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "expired_contracts": expired,
        "expiring_within_window": expiring,
        "upcoming_within_double_window": upcoming,
        "stale_contacts": stale,
        "average_engagement": round(avg_engagement, 2),
        "average_risk_score": round(avg_score, 2),
        "total_funding_commitment": round(total_funding, 2),
        "high_risk_funding": round(high_risk_funding, 2),
        "expired_funding": round(expired_funding, 2),
        "expiring_funding_within_window": round(expiring_funding, 2),
        "upcoming_funding_within_double_window": round(upcoming_funding, 2),
        "stale_contact_funding": round(stale_funding, 2),
        "average_value_at_risk": round(avg_value_risk, 2),
    }


def load_pg_dsn() -> str:
    dsn = os.environ.get("GS_PG_DSN")
    if dsn:
        return dsn
    host = os.environ.get("PGHOST")
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    port = os.environ.get("PGPORT")
    database = os.environ.get("PGDATABASE")
    if not host or not user or not password or not database:
        raise SystemExit(
            "Postgres export requires GS_PG_DSN or PGHOST/PGUSER/PGPASSWORD/PGDATABASE env vars."
        )
    port = port or "5432"
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def ensure_schema(cursor, schema: str) -> None:
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.renewal_runs (
            id SERIAL PRIMARY KEY,
            run_label TEXT,
            as_of DATE NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            renewal_window_days INTEGER NOT NULL,
            stale_contact_days INTEGER NOT NULL,
            low_engagement_threshold NUMERIC NOT NULL,
            high_issues_threshold INTEGER NOT NULL,
            weight_profile JSONB NOT NULL,
            total_partners INTEGER NOT NULL,
            high_risk INTEGER NOT NULL,
            medium_risk INTEGER NOT NULL,
            low_risk INTEGER NOT NULL,
            expired_contracts INTEGER NOT NULL,
            expiring_within_window INTEGER NOT NULL,
            upcoming_within_double_window INTEGER NOT NULL,
            stale_contacts INTEGER NOT NULL,
            average_engagement NUMERIC NOT NULL,
            average_risk_score NUMERIC NOT NULL,
            total_funding_commitment NUMERIC NOT NULL,
            high_risk_funding NUMERIC NOT NULL,
            expired_funding NUMERIC NOT NULL,
            expiring_funding_within_window NUMERIC NOT NULL,
            upcoming_funding_within_double_window NUMERIC NOT NULL,
            stale_contact_funding NUMERIC NOT NULL,
            average_value_at_risk NUMERIC NOT NULL
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.partner_scores (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES {schema}.renewal_runs(id) ON DELETE CASCADE,
            partner_id TEXT NOT NULL,
            partner_name TEXT NOT NULL,
            owner TEXT NOT NULL,
            last_contact_date DATE,
            contract_end_date DATE,
            engagement_score NUMERIC NOT NULL,
            meetings_last_90 INTEGER NOT NULL,
            referrals_last_90 INTEGER NOT NULL,
            issues_open INTEGER NOT NULL,
            funding_commitment NUMERIC NOT NULL,
            value_at_risk NUMERIC NOT NULL,
            days_since_contact INTEGER,
            days_to_contract_end INTEGER,
            risk_score INTEGER NOT NULL,
            risk_tier TEXT NOT NULL,
            expired BOOLEAN NOT NULL,
            reasons TEXT[] NOT NULL,
            action_code TEXT NOT NULL,
            action_note TEXT NOT NULL,
            action_priority INTEGER NOT NULL
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.action_queue (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES {schema}.renewal_runs(id) ON DELETE CASCADE,
            partner_id TEXT NOT NULL,
            partner_name TEXT NOT NULL,
            action_score INTEGER NOT NULL,
            action TEXT NOT NULL,
            focus TEXT NOT NULL,
            days_to_contract_end INTEGER,
            days_since_contact INTEGER,
            risk_score INTEGER NOT NULL,
            risk_tier TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.owner_summary (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES {schema}.renewal_runs(id) ON DELETE CASCADE,
            owner TEXT NOT NULL,
            total_partners INTEGER NOT NULL,
            high_risk INTEGER NOT NULL,
            medium_risk INTEGER NOT NULL,
            low_risk INTEGER NOT NULL,
            expired_contracts INTEGER NOT NULL,
            expiring_within_window INTEGER NOT NULL,
            stale_contacts INTEGER NOT NULL,
            average_risk_score NUMERIC NOT NULL,
            average_engagement NUMERIC NOT NULL,
            total_funding_commitment NUMERIC NOT NULL,
            value_at_risk NUMERIC NOT NULL
        )
        """
    )


def export_to_postgres(
    risks: List[PartnerRisk],
    actions: List[PartnerAction],
    owner_summary: List[Dict[str, float]],
    summary: Dict[str, object],
    as_of: date,
    args: argparse.Namespace,
) -> None:
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("psycopg is required for --export-postgres. Install with pip.") from exc

    dsn = load_pg_dsn()
    schema = os.environ.get("GS_PG_SCHEMA", "groupscholar_partner_renewal_tracker")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cursor:
            ensure_schema(cursor, schema)
            cursor.execute(
                f"""
                INSERT INTO {schema}.renewal_runs (
                    run_label, as_of, renewal_window_days, stale_contact_days,
                    low_engagement_threshold, high_issues_threshold, weight_profile,
                    total_partners, high_risk, medium_risk, low_risk, expired_contracts,
                    expiring_within_window, upcoming_within_double_window, stale_contacts,
                    average_engagement, average_risk_score, total_funding_commitment,
                    high_risk_funding, expired_funding, expiring_funding_within_window,
                    upcoming_funding_within_double_window, stale_contact_funding,
                    average_value_at_risk
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s
                )
                RETURNING id
                """,
                (
                    args.run_label,
                    as_of,
                    args.renewal_window_days,
                    args.stale_contact_days,
                    args.low_engagement_threshold,
                    args.high_issues_threshold,
                    json.dumps(summary["weight_profile"]),
                    summary["total_partners"],
                    summary["high_risk"],
                    summary["medium_risk"],
                    summary["low_risk"],
                    summary["expired_contracts"],
                    summary["expiring_within_window"],
                    summary["upcoming_within_double_window"],
                    summary["stale_contacts"],
                    summary["average_engagement"],
                    summary["average_risk_score"],
                    summary["total_funding_commitment"],
                    summary["high_risk_funding"],
                    summary["expired_funding"],
                    summary["expiring_funding_within_window"],
                    summary["upcoming_funding_within_double_window"],
                    summary["stale_contact_funding"],
                    summary["average_value_at_risk"],
                ),
            )
            run_id = cursor.fetchone()[0]

            cursor.executemany(
                f"""
                INSERT INTO {schema}.partner_scores (
                    run_id, partner_id, partner_name, owner, last_contact_date, contract_end_date,
                    engagement_score, meetings_last_90, referrals_last_90, issues_open,
                    funding_commitment, value_at_risk, days_since_contact, days_to_contract_end,
                    risk_score, risk_tier, expired, reasons, action_code, action_note, action_priority
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                """,
                [
                    (
                        run_id,
                        risk.record.partner_id,
                        risk.record.partner_name,
                        risk.record.owner,
                        risk.record.last_contact_date,
                        risk.record.contract_end_date,
                        risk.record.engagement_score,
                        risk.record.meetings_last_90,
                        risk.record.referrals_last_90,
                        risk.record.issues_open,
                        risk.record.funding_commitment,
                        compute_value_risk(risk),
                        risk.days_since_contact,
                        risk.days_to_contract_end,
                        risk.risk_score,
                        risk.risk_tier,
                        risk.expired,
                        risk.reasons,
                        risk.action_code,
                        risk.action_note,
                        risk.action_priority,
                    )
                    for risk in risks
                ],
            )

            cursor.executemany(
                f"""
                INSERT INTO {schema}.action_queue (
                    run_id, partner_id, partner_name, action_score, action, focus,
                    days_to_contract_end, days_since_contact, risk_score, risk_tier
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        run_id,
                        item.risk.record.partner_id,
                        item.risk.record.partner_name,
                        item.action_score,
                        item.action,
                        item.focus,
                        item.risk.days_to_contract_end,
                        item.risk.days_since_contact,
                        item.risk.risk_score,
                        item.risk.risk_tier,
                    )
                    for item in actions
                ],
            )

            cursor.executemany(
                f"""
                INSERT INTO {schema}.owner_summary (
                    run_id, owner, total_partners, high_risk, medium_risk, low_risk,
                    expired_contracts, expiring_within_window, stale_contacts,
                    average_risk_score, average_engagement, total_funding_commitment, value_at_risk
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        run_id,
                        owner["owner"],
                        owner["total_partners"],
                        owner["high_risk"],
                        owner["medium_risk"],
                        owner["low_risk"],
                        owner["expired_contracts"],
                        owner["expiring_within_window"],
                        owner["stale_contacts"],
                        owner["average_risk_score"],
                        owner["average_engagement"],
                        owner["total_funding_commitment"],
                        owner["value_at_risk"],
                    )
                    for owner in owner_summary
                ],
            )


def resolve_weights(
    profile: str,
    contact: Optional[float],
    contract: Optional[float],
    engagement: Optional[float],
    issues: Optional[float],
    meetings: Optional[float],
    referrals: Optional[float],
) -> Dict[str, float]:
    base = WEIGHT_PROFILES.get(profile, WEIGHT_PROFILES["balanced"]).copy()
    overrides = {
        "contact": contact,
        "contract": contract,
        "engagement": engagement,
        "issues": issues,
        "meetings": meetings,
        "referrals": referrals,
    }
    for key, value in overrides.items():
        if value is not None:
            base[key] = max(value, 0.0)
    return base


def build_owner_summary(
    risks: List[PartnerRisk],
    renewal_window_days: int,
    stale_contact_days: int,
) -> List[Dict[str, float]]:
    owners: Dict[str, List[PartnerRisk]] = {}
    for risk in risks:
        owners.setdefault(risk.record.owner, []).append(risk)

    snapshots: List[Dict[str, float]] = []
    for owner, items in owners.items():
        total = len(items)
        total_funding = sum(r.record.funding_commitment for r in items)
        value_at_risk = sum(compute_value_risk(r) for r in items)
        avg_risk = sum(r.risk_score for r in items) / total if total else 0.0
        avg_engagement = sum(r.record.engagement_score for r in items) / total if total else 0.0
        expiring = sum(
            1
            for r in items
            if r.days_to_contract_end is not None and r.days_to_contract_end <= renewal_window_days
        )
        expired = sum(1 for r in items if r.days_to_contract_end is not None and r.days_to_contract_end < 0)
        stale = sum(1 for r in items if r.days_since_contact is not None and r.days_since_contact >= stale_contact_days)
        high = sum(1 for r in items if r.risk_tier == "high")
        medium = sum(1 for r in items if r.risk_tier == "medium")
        low = sum(1 for r in items if r.risk_tier == "low")
        snapshots.append(
            {
                "owner": owner,
                "total_partners": total,
                "high_risk": high,
                "medium_risk": medium,
                "low_risk": low,
                "expired_contracts": expired,
                "expiring_within_window": expiring,
                "stale_contacts": stale,
                "average_risk_score": round(avg_risk, 2),
                "average_engagement": round(avg_engagement, 2),
                "total_funding_commitment": round(total_funding, 2),
                "value_at_risk": round(value_at_risk, 2),
            }
        )
    snapshots.sort(
        key=lambda s: (s["value_at_risk"], s["average_risk_score"], s["total_funding_commitment"]),
        reverse=True,
    )
    return snapshots


def build_reason_summary(risks: List[PartnerRisk]) -> List[Dict[str, float]]:
    total = len(risks)
    summary: Dict[str, Dict[str, float]] = {}
    for risk in risks:
        reasons = risk.reasons or ["none"]
        for reason in reasons:
            bucket = summary.setdefault(
                reason,
                {
                    "reason": reason,
                    "total_partners": 0,
                    "high_risk_partners": 0,
                    "total_funding_commitment": 0.0,
                    "value_at_risk": 0.0,
                },
            )
            bucket["total_partners"] += 1
            if risk.risk_tier == "high":
                bucket["high_risk_partners"] += 1
            bucket["total_funding_commitment"] = round(
                float(bucket["total_funding_commitment"]) + risk.record.funding_commitment,
                2,
            )
            bucket["value_at_risk"] = round(
                float(bucket["value_at_risk"]) + compute_value_risk(risk),
                2,
            )

    rows = list(summary.values())
    for row in rows:
        percent = (row["total_partners"] / total * 100.0) if total else 0.0
        row["percent_of_partners"] = round(percent, 2)
    rows.sort(
        key=lambda r: (r["value_at_risk"], r["total_partners"], r["reason"]),
        reverse=True,
    )
    return rows


def build_action_plan(
    risk: PartnerRisk,
    renewal_window_days: int,
    stale_contact_days: int,
    low_engagement_threshold: float,
    high_issues_threshold: int,
) -> PartnerAction:
    action_score = risk.risk_score
    action = "Monitor relationship"
    focus = "monitor"

    if risk.days_to_contract_end is not None and risk.days_to_contract_end < 0:
        action_score += 30
        action = "Escalate renewal recovery"
        focus = "renewal_overdue"
    elif risk.days_to_contract_end is not None and risk.days_to_contract_end <= renewal_window_days:
        action_score += 20
        action = "Launch renewal plan"
        focus = "renewal_due"
    elif risk.days_to_contract_end is None:
        action_score += 15
        action = "Confirm contract end date"
        focus = "missing_contract"
    elif risk.days_to_contract_end is not None and risk.days_to_contract_end <= renewal_window_days * 2:
        action_score += 8
        action = "Prep renewal runway"
        focus = "renewal_horizon"

    if risk.days_since_contact is None:
        action_score += 15
        if focus in {"monitor", "renewal_horizon", "missing_contract"}:
            action = "Log last contact + schedule check-in"
            focus = "missing_contact"
    elif risk.days_since_contact >= stale_contact_days:
        action_score += 15
        if focus in {"monitor", "renewal_horizon", "missing_contract"}:
            action = "Schedule relationship reset"
            focus = "stale_contact"

    if risk.record.engagement_score < low_engagement_threshold:
        action_score += 10
        if focus in {"monitor", "renewal_horizon"}:
            action = "Deploy engagement plan"
            focus = "low_engagement"

    if risk.record.issues_open >= high_issues_threshold:
        action_score += 8
        if focus in {"monitor", "renewal_horizon"}:
            action = "Resolve open issues"
            focus = "issue_resolution"

    action_score = min(action_score, 100)

    return PartnerAction(
        risk=risk,
        action_score=action_score,
        action=action,
        focus=focus,
    )


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def html_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def month_bucket_label(year: int, month: int) -> str:
    return date(year, month, 1).strftime("%b %Y")


def build_renewal_calendar(
    risks: List[PartnerRisk],
    as_of: date,
    months: int,
) -> List[Dict[str, object]]:
    if months <= 0:
        return []

    buckets: Dict[str, Dict[str, object]] = {}
    overdue_key = "overdue"
    missing_key = "missing_contract"

    def init_bucket(bucket: str, label: str) -> None:
        if bucket not in buckets:
            buckets[bucket] = {
                "bucket": bucket,
                "label": label,
                "expiring_partners": 0,
                "expiring_funding": 0.0,
                "high_risk_partners": 0,
                "high_risk_funding": 0.0,
            }

    init_bucket(overdue_key, "Overdue")
    init_bucket(missing_key, "Missing contract date")

    months_list: List[Tuple[int, int]] = []
    year = as_of.year
    month = as_of.month
    for _ in range(months):
        months_list.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    for year, month in months_list:
        key = f"{year:04d}-{month:02d}"
        init_bucket(key, month_bucket_label(year, month))

    for risk in risks:
        contract_end = risk.record.contract_end_date
        if contract_end is None:
            bucket_key = missing_key
        elif contract_end < as_of:
            bucket_key = overdue_key
        else:
            bucket_key = f"{contract_end.year:04d}-{contract_end.month:02d}"
            if bucket_key not in buckets:
                continue
        bucket = buckets[bucket_key]
        bucket["expiring_partners"] += 1
        bucket["expiring_funding"] = round(
            float(bucket["expiring_funding"]) + risk.record.funding_commitment, 2
        )
        if risk.risk_tier == "high":
            bucket["high_risk_partners"] += 1
            bucket["high_risk_funding"] = round(
                float(bucket["high_risk_funding"]) + risk.record.funding_commitment, 2
            )

    ordered: List[Dict[str, object]] = []
    ordered.append(buckets[overdue_key])
    for year, month in months_list:
        ordered.append(buckets[f"{year:04d}-{month:02d}"])
    ordered.append(buckets[missing_key])
    return ordered


def write_html_report(
    path: str,
    as_of: date,
    summary: Dict[str, float],
    top: List[PartnerRisk],
    top_value: List[PartnerRisk],
    actions: List[PartnerAction],
    owners: List[Dict[str, float]],
    calendar: List[Dict[str, object]],
    warnings: List[str],
) -> None:
    if not path:
        return
    rows_top = "\n".join(
        f"<tr><td>{html_escape(r.record.partner_name)}</td>"
        f"<td>{html_escape(r.record.partner_id)}</td>"
        f"<td>{r.risk_score}</td>"
        f"<td>{html_escape(r.risk_tier)}</td>"
        f"<td>{'' if r.days_to_contract_end is None else r.days_to_contract_end}</td>"
        f"<td>{'' if r.days_since_contact is None else r.days_since_contact}</td>"
        f"<td>{html_escape(', '.join(r.reasons))}</td></tr>"
        for r in top
    )
    rows_value = "\n".join(
        f"<tr><td>{html_escape(r.record.partner_name)}</td>"
        f"<td>{html_escape(r.record.partner_id)}</td>"
        f"<td>{r.record.funding_commitment}</td>"
        f"<td>{compute_value_risk(r)}</td>"
        f"<td>{r.risk_score}</td>"
        f"<td>{html_escape(r.risk_tier)}</td></tr>"
        for r in top_value
    )
    rows_actions = "\n".join(
        f"<tr><td>{html_escape(a.risk.record.partner_name)}</td>"
        f"<td>{html_escape(a.risk.record.partner_id)}</td>"
        f"<td>{a.action_score}</td>"
        f"<td>{html_escape(a.action)}</td>"
        f"<td>{html_escape(a.focus)}</td>"
        f"<td>{'' if a.risk.days_to_contract_end is None else a.risk.days_to_contract_end}</td>"
        f"<td>{'' if a.risk.days_since_contact is None else a.risk.days_since_contact}</td>"
        f"<td>{a.risk.risk_score}</td></tr>"
        for a in actions
    )
    rows_owners = "\n".join(
        f"<tr><td>{html_escape(o['owner'])}</td>"
        f"<td>{o['total_partners']}</td>"
        f"<td>{o['high_risk']}</td>"
        f"<td>{o['medium_risk']}</td>"
        f"<td>{o['low_risk']}</td>"
        f"<td>{o['expiring_within_window']}</td>"
        f"<td>{o['stale_contacts']}</td>"
        f"<td>{o['average_risk_score']}</td>"
        f"<td>{o['value_at_risk']}</td></tr>"
        for o in owners
    )
    rows_calendar = "\n".join(
        f"<tr><td>{html_escape(item['label'])}</td>"
        f"<td>{item['expiring_partners']}</td>"
        f"<td>{item['expiring_funding']}</td>"
        f"<td>{item['high_risk_partners']}</td>"
        f"<td>{item['high_risk_funding']}</td></tr>"
        for item in calendar
    )
    calendar_chart = build_calendar_chart_html(calendar)
    warning_list = ""
    if warnings:
        warning_items = "\n".join(f"<li>{html_escape(item)}</li>" for item in warnings)
        warning_list = f"<section><h2>Warnings</h2><ul>{warning_items}</ul></section>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Partner Renewal Tracker Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f1f1f;
      --muted: #4a4a4a;
      --panel: #f3f4f6;
      --accent: #1f6feb;
      --border: #d5d7db;
    }}
    body {{
      font-family: "Georgia", "Times New Roman", serif;
      color: var(--ink);
      margin: 0;
      background: #faf9f6;
    }}
    header {{
      padding: 32px 40px 12px;
      background: linear-gradient(120deg, #fff8e8, #f2f7ff);
      border-bottom: 1px solid var(--border);
    }}
    header h1 {{
      margin: 0;
      font-size: 30px;
    }}
    header p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    main {{
      padding: 28px 40px 48px;
    }}
    section {{
      margin-bottom: 28px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 20px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .summary-card span {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }}
    .summary-card strong {{
      font-size: 20px;
      display: block;
      margin-top: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: #f8f7f3;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }}
    .note {{
      font-size: 12px;
      color: var(--muted);
    }}
    .chart {{
      margin: 12px 0 18px;
      padding: 12px;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
    }}
    .chart svg {{
      width: 100%;
      height: auto;
    }}
    .chart-legend {{
      display: flex;
      gap: 12px;
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }}
    .legend-swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 6px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Partner Renewal Tracker</h1>
    <p>Report date: {as_of.strftime(DATE_FMT)}</p>
  </header>
  <main>
    <section>
      <h2>Summary</h2>
      <div class="summary-grid">
        <div class="summary-card"><span>Total partners</span><strong>{summary['total_partners']}</strong></div>
        <div class="summary-card"><span>High risk</span><strong>{summary['high_risk']}</strong></div>
        <div class="summary-card"><span>Medium risk</span><strong>{summary['medium_risk']}</strong></div>
        <div class="summary-card"><span>Low risk</span><strong>{summary['low_risk']}</strong></div>
        <div class="summary-card"><span>Expired contracts</span><strong>{summary['expired_contracts']}</strong></div>
        <div class="summary-card"><span>Expiring window</span><strong>{summary['expiring_within_window']}</strong></div>
        <div class="summary-card"><span>Stale contacts</span><strong>{summary['stale_contacts']}</strong></div>
        <div class="summary-card"><span>Total funding</span><strong>{summary['total_funding_commitment']}</strong></div>
        <div class="summary-card"><span>High-risk funding</span><strong>{summary['high_risk_funding']}</strong></div>
        <div class="summary-card"><span>Avg risk score</span><strong>{summary['average_risk_score']}</strong></div>
        <div class="summary-card"><span>Avg value at risk</span><strong>{summary['average_value_at_risk']}</strong></div>
        <div class="summary-card"><span>Avg engagement</span><strong>{summary['average_engagement']}</strong></div>
      </div>
      <p class="note">Weight profile: {html_escape(summary['weight_profile'])}</p>
    </section>

    <section>
      <h2>Top at-risk partners</h2>
      <table>
        <thead>
          <tr>
            <th>Partner</th>
            <th>ID</th>
            <th>Score</th>
            <th>Tier</th>
            <th>Days to end</th>
            <th>Days since contact</th>
            <th>Reasons</th>
          </tr>
        </thead>
        <tbody>
          {rows_top or '<tr><td colspan="7">No partners found.</td></tr>'}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Top value at risk</h2>
      <table>
        <thead>
          <tr>
            <th>Partner</th>
            <th>ID</th>
            <th>Commitment</th>
            <th>Value at risk</th>
            <th>Score</th>
            <th>Tier</th>
          </tr>
        </thead>
        <tbody>
          {rows_value or '<tr><td colspan="6">No partners found.</td></tr>'}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Action queue</h2>
      <table>
        <thead>
          <tr>
            <th>Partner</th>
            <th>ID</th>
            <th>Action score</th>
            <th>Action</th>
            <th>Focus</th>
            <th>Days to end</th>
            <th>Days since contact</th>
            <th>Risk score</th>
          </tr>
        </thead>
        <tbody>
          {rows_actions or '<tr><td colspan="8">No actions found.</td></tr>'}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Owner snapshot</h2>
      <table>
        <thead>
          <tr>
            <th>Owner</th>
            <th>Partners</th>
            <th>High</th>
            <th>Medium</th>
            <th>Low</th>
            <th>Expiring</th>
            <th>Stale</th>
            <th>Avg risk</th>
            <th>Value at risk</th>
          </tr>
        </thead>
        <tbody>
          {rows_owners or '<tr><td colspan="9">No owners found.</td></tr>'}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Renewal calendar</h2>
      {calendar_chart}
      <table>
        <thead>
          <tr>
            <th>Bucket</th>
            <th>Partners</th>
            <th>Funding</th>
            <th>High-risk partners</th>
            <th>High-risk funding</th>
          </tr>
        </thead>
        <tbody>
          {rows_calendar or '<tr><td colspan="5">No calendar data.</td></tr>'}
        </tbody>
      </table>
    </section>
    {warning_list}
  </main>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)


def build_calendar_chart_html(calendar: List[Dict[str, object]]) -> str:
    if not calendar:
        return "<p class=\"note\">No calendar data available.</p>"

    values = [
        max(float(item.get("expiring_funding", 0.0)), float(item.get("high_risk_funding", 0.0)))
        for item in calendar
    ]
    max_value = max(values) if values else 0.0
    if max_value <= 0:
        return "<p class=\"note\">No funding values to chart yet.</p>"

    width = 680
    height = 240
    padding = 36
    baseline = height - padding
    chart_width = width - padding * 2
    chart_height = height - padding * 2
    bucket_width = chart_width / max(len(calendar), 1)
    bar_width = max(bucket_width * 0.28, 10)
    scale = chart_height / max_value if max_value else 0.0

    def label_for(item: Dict[str, object]) -> str:
        bucket = str(item.get("bucket", ""))
        label = str(item.get("label", ""))
        if bucket == "overdue":
            return "Overdue"
        if bucket == "missing_contract":
            return "Missing"
        if len(label) > 9:
            return label.split()[0]
        return label

    bars = []
    labels = []
    for idx, item in enumerate(calendar):
        expiring = float(item.get("expiring_funding", 0.0))
        high_risk = float(item.get("high_risk_funding", 0.0))
        x_center = padding + bucket_width * idx + bucket_width / 2
        exp_height = expiring * scale
        high_height = high_risk * scale
        exp_x = x_center - bar_width - 2
        high_x = x_center + 2
        bars.append(
            f'<rect x="{exp_x:.1f}" y="{baseline - exp_height:.1f}" '
            f'width="{bar_width:.1f}" height="{exp_height:.1f}" fill="#1f6feb" opacity="0.85"></rect>'
        )
        bars.append(
            f'<rect x="{high_x:.1f}" y="{baseline - high_height:.1f}" '
            f'width="{bar_width:.1f}" height="{high_height:.1f}" fill="#f97316" opacity="0.85"></rect>'
        )
        labels.append(
            f'<text x="{x_center:.1f}" y="{height - 10}" font-size="10" '
            f'text-anchor="middle" fill="#4a4a4a">{html_escape(label_for(item))}</text>'
        )

    svg = (
        f'<div class="chart">'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Renewal funding by bucket">'
        f'<line x1="{padding}" y1="{baseline}" x2="{width - padding}" y2="{baseline}" stroke="#d5d7db" />'
        f'{"".join(bars)}'
        f'{"".join(labels)}'
        f'</svg>'
        f'<div class="chart-legend">'
        f'<span><span class="legend-swatch" style="background:#1f6feb"></span>Expiring funding</span>'
        f'<span><span class="legend-swatch" style="background:#f97316"></span>High-risk funding</span>'
        f'</div>'
        f'</div>'
    )
    return svg


def render_console(
    summary: Dict[str, float],
    top: List[PartnerRisk],
    top_value: List[PartnerRisk],
    actions: List[PartnerAction],
    owners: List[Dict[str, float]],
    calendar: List[Dict[str, object]],
    top_owners: int,
    calendar_months: int,
) -> None:
    print("Partner Renewal Tracker")
    print("=" * 26)
    print(f"Total partners: {summary['total_partners']}")
    print(f"Risk mix: high {summary['high_risk']} | medium {summary['medium_risk']} | low {summary['low_risk']}")
    print(f"Expired contracts: {summary['expired_contracts']}")
    print(f"Expiring within window: {summary['expiring_within_window']}")
    print(f"Upcoming within 2x window: {summary['upcoming_within_double_window']}")
    print(f"Stale contacts: {summary['stale_contacts']}")
    print(f"Average engagement: {summary['average_engagement']}")
    print(f"Average risk score: {summary['average_risk_score']}")
    print(f"Total funding commitment: {summary['total_funding_commitment']}")
    print(f"High-risk funding: {summary['high_risk_funding']}")
    print(f"Expired funding: {summary['expired_funding']}")
    print(f"Expiring funding within window: {summary['expiring_funding_within_window']}")
    print(f"Upcoming funding within 2x window: {summary['upcoming_funding_within_double_window']}")
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
    if actions:
        print("\nAction queue")
        print("-" * 26)
        for item in actions:
            record = item.risk.record
            days_to_end = "n/a" if item.risk.days_to_contract_end is None else str(item.risk.days_to_contract_end)
            days_since = "n/a" if item.risk.days_since_contact is None else str(item.risk.days_since_contact)
            print(
                f"{record.partner_name} ({record.partner_id}) | action_score {item.action_score} | "
                f"{item.action} | focus {item.focus} | days_to_end {days_to_end} | days_since_contact {days_since}"
            )
    if owners:
        print("\nOwner risk snapshot")
        print("-" * 26)
        for owner in owners[:top_owners]:
            print(
                f"{owner['owner']} | partners {owner['total_partners']} | high {owner['high_risk']} | "
                f"expiring {owner['expiring_within_window']} | value_at_risk {owner['value_at_risk']} | "
                f"avg_risk {owner['average_risk_score']}"
            )
    if calendar:
        print(f"\nRenewal calendar (next {calendar_months} months)")
        print("-" * 26)
        for item in calendar:
            print(
                f"{item['label']} | partners {item['expiring_partners']} | funding {item['expiring_funding']} | "
                f"high_risk {item['high_risk_partners']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Track partner renewal risk signals from CSV exports.")
    parser.add_argument("--input", required=True, help="Path to partner CSV export.")
    parser.add_argument("--as-of", help="Override as-of date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--json-out", help="Optional path for JSON report output.")
    parser.add_argument("--html-out", help="Optional path for HTML report output.")
    parser.add_argument("--csv-out", help="Optional path for partner risk CSV export.")
    parser.add_argument("--actions-csv-out", help="Optional path for action queue CSV export.")
    parser.add_argument("--owners-csv-out", help="Optional path for owner summary CSV export.")
    parser.add_argument("--calendar-csv-out", help="Optional path for renewal calendar CSV export.")
    parser.add_argument("--export-postgres", action="store_true", help="Write results to Postgres.")
    parser.add_argument("--run-label", help="Optional label to store with Postgres run.")
    parser.add_argument("--top", type=int, default=10, help="How many top at-risk partners to show.")
    parser.add_argument("--top-value", type=int, default=5, help="How many top value-at-risk partners to show.")
    parser.add_argument("--top-actions", type=int, default=8, help="How many action queue partners to show.")
    parser.add_argument("--top-owners", type=int, default=6, help="How many owners to list in snapshot.")
    parser.add_argument("--calendar-months", type=int, default=6, help="Months to include in renewal calendar.")
    parser.add_argument("--stale-contact-days", type=int, default=45)
    parser.add_argument("--renewal-window-days", type=int, default=90)
    parser.add_argument("--low-engagement-threshold", type=float, default=55.0)
    parser.add_argument("--high-issues-threshold", type=int, default=3)
    parser.add_argument("--weight-profile", default="balanced", choices=sorted(WEIGHT_PROFILES))
    parser.add_argument("--weight-contact", type=float)
    parser.add_argument("--weight-contract", type=float)
    parser.add_argument("--weight-engagement", type=float)
    parser.add_argument("--weight-issues", type=float)
    parser.add_argument("--weight-meetings", type=float)
    parser.add_argument("--weight-referrals", type=float)

    args = parser.parse_args()

    as_of = date.today()
    if args.as_of:
        parsed = parse_date(args.as_of)
        if not parsed:
            raise SystemExit("Invalid --as-of date. Use YYYY-MM-DD.")
        as_of = parsed

    weights = resolve_weights(
        args.weight_profile,
        args.weight_contact,
        args.weight_contract,
        args.weight_engagement,
        args.weight_issues,
        args.weight_meetings,
        args.weight_referrals,
    )

    records, warnings = load_partners(args.input)
    risks = [
        compute_risk(
            record,
            as_of,
            args.stale_contact_days,
            args.renewal_window_days,
            args.low_engagement_threshold,
            args.high_issues_threshold,
            weights,
        )
        for record in records
    ]

    risks.sort(key=lambda r: (r.risk_score, r.record.partner_name), reverse=True)
    value_ranked = sorted(
        risks,
        key=lambda r: (compute_value_risk(r), r.risk_score, r.record.partner_name),
        reverse=True,
    )
    actions = [
        build_action_plan(
            risk,
            args.renewal_window_days,
            args.stale_contact_days,
            args.low_engagement_threshold,
            args.high_issues_threshold,
        )
        for risk in risks
    ]
    actions.sort(key=lambda item: (item.action_score, item.risk.risk_score, item.risk.record.partner_name), reverse=True)
    summary = summarize(risks, args.renewal_window_days, args.stale_contact_days, weights)
    owner_summary = build_owner_summary(risks, args.renewal_window_days, args.stale_contact_days)
    renewal_calendar = build_renewal_calendar(risks, as_of, args.calendar_months)

    render_console(
        summary,
        risks[: args.top],
        value_ranked[: args.top_value],
        actions[: args.top_actions],
        owner_summary,
        renewal_calendar,
        args.top_owners,
        args.calendar_months,
    )

    if warnings:
        print("\nWarnings")
        print("-" * 26)
        for warning in warnings:
            print(warning)

    if args.json_out:
        payload = {
            "as_of": as_of.strftime(DATE_FMT),
            "summary": summary,
            "owner_summary": owner_summary,
            "renewal_calendar": renewal_calendar,
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
            "action_queue": [
                {
                    "partner_id": item.risk.record.partner_id,
                    "partner_name": item.risk.record.partner_name,
                    "action_score": item.action_score,
                    "action": item.action,
                    "focus": item.focus,
                    "days_to_contract_end": item.risk.days_to_contract_end,
                    "days_since_contact": item.risk.days_since_contact,
                    "risk_score": item.risk.risk_score,
                    "risk_tier": item.risk.risk_tier,
                }
                for item in actions
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    if args.html_out:
        write_html_report(
            args.html_out,
            as_of,
            summary,
            risks[: args.top],
            value_ranked[: args.top_value],
            actions[: args.top_actions],
            owner_summary[: args.top_owners],
            renewal_calendar,
            warnings,
        )

    if args.csv_out:
        partner_rows = [
            {
                "partner_id": risk.record.partner_id,
                "partner_name": risk.record.partner_name,
                "owner": risk.record.owner,
                "last_contact_date": risk.record.last_contact_date.strftime(DATE_FMT)
                if risk.record.last_contact_date
                else "",
                "contract_end_date": risk.record.contract_end_date.strftime(DATE_FMT)
                if risk.record.contract_end_date
                else "",
                "engagement_score": risk.record.engagement_score,
                "meetings_last_90": risk.record.meetings_last_90,
                "referrals_last_90": risk.record.referrals_last_90,
                "issues_open": risk.record.issues_open,
                "funding_commitment": risk.record.funding_commitment,
                "value_at_risk": compute_value_risk(risk),
                "days_since_contact": risk.days_since_contact if risk.days_since_contact is not None else "",
                "days_to_contract_end": risk.days_to_contract_end if risk.days_to_contract_end is not None else "",
                "risk_score": risk.risk_score,
                "risk_tier": risk.risk_tier,
                "expired": risk.expired,
                "reasons": "|".join(risk.reasons),
                "action_code": risk.action_code,
                "action_note": risk.action_note,
                "action_priority": risk.action_priority,
            }
            for risk in risks
        ]
        write_csv(
            args.csv_out,
            partner_rows,
            [
                "partner_id",
                "partner_name",
                "owner",
                "last_contact_date",
                "contract_end_date",
                "engagement_score",
                "meetings_last_90",
                "referrals_last_90",
                "issues_open",
                "funding_commitment",
                "value_at_risk",
                "days_since_contact",
                "days_to_contract_end",
                "risk_score",
                "risk_tier",
                "expired",
                "reasons",
                "action_code",
                "action_note",
                "action_priority",
            ],
        )

    if args.actions_csv_out:
        action_rows = [
            {
                "partner_id": item.risk.record.partner_id,
                "partner_name": item.risk.record.partner_name,
                "owner": item.risk.record.owner,
                "action_score": item.action_score,
                "action": item.action,
                "focus": item.focus,
                "days_to_contract_end": item.risk.days_to_contract_end
                if item.risk.days_to_contract_end is not None
                else "",
                "days_since_contact": item.risk.days_since_contact
                if item.risk.days_since_contact is not None
                else "",
                "risk_score": item.risk.risk_score,
                "risk_tier": item.risk.risk_tier,
            }
            for item in actions
        ]
        write_csv(
            args.actions_csv_out,
            action_rows,
            [
                "partner_id",
                "partner_name",
                "owner",
                "action_score",
                "action",
                "focus",
                "days_to_contract_end",
                "days_since_contact",
                "risk_score",
                "risk_tier",
            ],
        )

    if args.owners_csv_out:
        owner_rows = [
            {
                "owner": owner["owner"],
                "total_partners": owner["total_partners"],
                "high_risk": owner["high_risk"],
                "medium_risk": owner["medium_risk"],
                "low_risk": owner["low_risk"],
                "expired_contracts": owner["expired_contracts"],
                "expiring_within_window": owner["expiring_within_window"],
                "stale_contacts": owner["stale_contacts"],
                "average_risk_score": owner["average_risk_score"],
                "average_engagement": owner["average_engagement"],
                "total_funding_commitment": owner["total_funding_commitment"],
                "value_at_risk": owner["value_at_risk"],
            }
            for owner in owner_summary
        ]
        write_csv(
            args.owners_csv_out,
            owner_rows,
            [
                "owner",
                "total_partners",
                "high_risk",
                "medium_risk",
                "low_risk",
                "expired_contracts",
                "expiring_within_window",
                "stale_contacts",
                "average_risk_score",
                "average_engagement",
                "total_funding_commitment",
                "value_at_risk",
            ],
        )

    if args.calendar_csv_out:
        calendar_rows = [
            {
                "bucket": item["bucket"],
                "label": item["label"],
                "expiring_partners": item["expiring_partners"],
                "expiring_funding": item["expiring_funding"],
                "high_risk_partners": item["high_risk_partners"],
                "high_risk_funding": item["high_risk_funding"],
            }
            for item in renewal_calendar
        ]
        write_csv(
            args.calendar_csv_out,
            calendar_rows,
            [
                "bucket",
                "label",
                "expiring_partners",
                "expiring_funding",
                "high_risk_partners",
                "high_risk_funding",
            ],
        )

    if args.export_postgres:
        export_to_postgres(risks, actions, owner_summary, summary, as_of, args)


if __name__ == "__main__":
    main()
