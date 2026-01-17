#!/usr/bin/env python3
"""
Generate Test Data for Talki Session Logs

Creates realistic session log data and uploads to S3 for Athena analysis.
Simulates diverse usage patterns across languages, regions, and time periods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import boto3

# Configuration
S3_BUCKET = "talki-metrics-data-eu-west-1"
S3_PREFIX = "session_completed"
AWS_REGION = "eu-west-1"

# Realistic distributions based on Talki usage patterns
LANGUAGES = {
    "en-GB": 0.25,  # UK English
    "en-US": 0.15,  # US English
    "es-ES": 0.15,  # Spanish
    "fr-FR": 0.12,  # French
    "de-DE": 0.10,  # German
    "it-IT": 0.08,  # Italian
    "pt-BR": 0.06,  # Portuguese
    "nl-NL": 0.04,  # Dutch
    "pl-PL": 0.03,  # Polish
    "ja-JP": 0.02,  # Japanese
}

REGIONS = {
    "eu-west-1": 0.35,
    "eu-west-2": 0.20,
    "us-east-1": 0.20,
    "us-west-2": 0.10,
    "ap-northeast-1": 0.08,
    "sa-east-1": 0.07,
}

MODELS = [
    {"provider": "anthropic", "name": "claude-3-haiku-20240307", "weight": 0.60, "input_price": 0.00025, "output_price": 0.00125},
    {"provider": "anthropic", "name": "claude-3-sonnet-20240229", "weight": 0.25, "input_price": 0.003, "output_price": 0.015},
    {"provider": "anthropic", "name": "claude-3-opus-20240229", "weight": 0.10, "input_price": 0.015, "output_price": 0.075},
    {"provider": "openai", "name": "gpt-4o-mini", "weight": 0.05, "input_price": 0.00015, "output_price": 0.0006},
]

SOURCES = ["mobile_app", "web_app", "alexa_skill", "google_assistant"]

ERROR_CODES = [
    None, None, None, None, None, None, None, None, None,  # 90% success
    "TIMEOUT", "RATE_LIMIT", "CONTENT_FILTER", "MODEL_ERROR"
]


@dataclass
class SessionLog:
    """A single session log entry."""
    session_id: str
    family_id_hash: str
    child_id_hash: str
    language: str
    region: str
    source: str
    stage: str
    timestamp: str
    duration_seconds: int
    latency_ms: int
    cost_usd: float
    success: bool
    error_code: str | None
    model_provider: str
    model_name: str
    input_tokens: int
    output_tokens: int
    year: str
    month: str
    day: str

    def to_json(self) -> str:
        data = asdict(self)
        # Convert None to empty string for JSON
        if data["error_code"] is None:
            data["error_code"] = ""
        # Remove partition keys from JSON (they're in the path)
        del data["year"]
        del data["month"]
        del data["day"]
        return json.dumps(data)


def weighted_choice(options: dict[str, float]) -> str:
    """Choose from options based on weights."""
    items = list(options.keys())
    weights = list(options.values())
    return random.choices(items, weights=weights, k=1)[0]


def generate_family_id() -> str:
    """Generate a hashed family ID."""
    raw_id = f"family_{random.randint(1000, 9999)}_{random.randint(1, 100)}"
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]


def generate_child_id(family_id: str) -> str:
    """Generate a hashed child ID for a family."""
    raw_id = f"child_{family_id}_{random.randint(1, 5)}"
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]


def generate_session_id() -> str:
    """Generate a unique session ID."""
    import uuid
    return str(uuid.uuid4())


def generate_session(
    timestamp: datetime,
    families: dict[str, list[str]],
    stage: str = "prod",
) -> SessionLog:
    """Generate a realistic session log entry."""

    # Generate session ID
    session_id = generate_session_id()

    # Pick or create a family
    if random.random() < 0.3 or not families:  # 30% new families
        family_id = generate_family_id()
        child_ids = [generate_child_id(family_id) for _ in range(random.randint(1, 3))]
        families[family_id] = child_ids
    else:
        family_id = random.choice(list(families.keys()))

    child_id = random.choice(families[family_id])

    # Session properties
    language = weighted_choice(LANGUAGES)
    region = weighted_choice(REGIONS)
    source = random.choice(SOURCES)

    # Model selection
    model = random.choices(MODELS, weights=[m["weight"] for m in MODELS], k=1)[0]

    # Token counts (varies by conversation complexity)
    complexity = random.choice(["simple", "medium", "complex"])
    if complexity == "simple":
        input_tokens = random.randint(50, 200)
        output_tokens = random.randint(30, 150)
    elif complexity == "medium":
        input_tokens = random.randint(200, 500)
        output_tokens = random.randint(100, 300)
    else:
        input_tokens = random.randint(500, 1500)
        output_tokens = random.randint(200, 800)

    # Calculate cost
    cost_usd = (input_tokens / 1000 * model["input_price"]) + (output_tokens / 1000 * model["output_price"])

    # Duration and latency
    duration_seconds = random.randint(5, 300)  # 5 seconds to 5 minutes
    latency_ms = random.randint(200, 2000) + (1000 if model["name"].startswith("claude-3-opus") else 0)

    # Error handling
    error_code = random.choice(ERROR_CODES)
    success = error_code is None

    return SessionLog(
        session_id=session_id,
        family_id_hash=family_id,
        child_id_hash=child_id,
        language=language,
        region=region,
        source=source,
        stage=stage,
        timestamp=timestamp.isoformat() + "Z",
        duration_seconds=duration_seconds,
        latency_ms=latency_ms,
        cost_usd=round(cost_usd, 6),
        success=success,
        error_code=error_code,
        model_provider=model["provider"],
        model_name=model["name"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        year=timestamp.strftime("%Y"),
        month=timestamp.strftime("%m"),
        day=timestamp.strftime("%d"),
    )


def generate_sessions(
    num_sessions: int,
    start_date: datetime,
    end_date: datetime,
    stage: str = "prod",
) -> list[SessionLog]:
    """Generate multiple session logs across a date range."""

    sessions = []
    families: dict[str, list[str]] = {}

    # Generate timestamps spread across the date range
    date_range_seconds = (end_date - start_date).total_seconds()

    for _ in range(num_sessions):
        # Random timestamp within range
        random_seconds = random.uniform(0, date_range_seconds)
        timestamp = start_date + timedelta(seconds=random_seconds)

        session = generate_session(timestamp, families, stage)
        sessions.append(session)

    # Sort by timestamp
    sessions.sort(key=lambda s: s.timestamp)

    return sessions


def upload_to_s3(
    sessions: list[SessionLog],
    bucket: str,
    prefix: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Upload sessions to S3, partitioned by year/month/day."""

    # Group by partition
    partitions: dict[str, list[SessionLog]] = {}
    for session in sessions:
        key = f"{session.year}/{session.month}/{session.day}"
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(session)

    s3 = boto3.client("s3", region_name=AWS_REGION) if not dry_run else None

    stats = {"files_uploaded": 0, "total_records": 0}

    for partition_key, partition_sessions in partitions.items():
        # Create NDJSON content
        content = "\n".join(s.to_json() for s in partition_sessions)

        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{prefix}/year={partition_key.split('/')[0]}/month={partition_key.split('/')[1]}/day={partition_key.split('/')[2]}/generated_{timestamp}.json"

        if dry_run:
            print(f"  [DRY RUN] Would upload {len(partition_sessions)} records to s3://{bucket}/{s3_key}")
        else:
            s3.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=content.encode("utf-8"),
                ContentType="application/json",
            )
            print(f"  Uploaded {len(partition_sessions)} records to s3://{bucket}/{s3_key}")

        stats["files_uploaded"] += 1
        stats["total_records"] += len(partition_sessions)

    return stats


def preview_sessions(sessions: list[SessionLog], limit: int = 5) -> None:
    """Print a preview of generated sessions."""
    print("\n  Sample Generated Sessions:")
    print("  " + "─" * 70)

    for session in sessions[:limit]:
        print(f"    {session.timestamp[:19]} | {session.language:5} | {session.model_name:30} | ${session.cost_usd:.4f}")

    if len(sessions) > limit:
        print(f"    ... and {len(sessions) - limit} more sessions")


def print_statistics(sessions: list[SessionLog]) -> None:
    """Print statistics about generated data."""
    print("\n  Generated Data Statistics:")
    print("  " + "─" * 70)

    # Basic counts
    total_cost = sum(s.cost_usd for s in sessions)
    success_count = sum(1 for s in sessions if s.success)

    print(f"    Total sessions:      {len(sessions):,}")
    print(f"    Total cost:          ${total_cost:,.4f}")
    print(f"    Success rate:        {success_count / len(sessions) * 100:.1f}%")

    # By language
    lang_counts: dict[str, int] = {}
    for s in sessions:
        lang_counts[s.language] = lang_counts.get(s.language, 0) + 1

    print(f"\n    Languages:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"      {lang}: {count:,} ({count / len(sessions) * 100:.1f}%)")

    # By model
    model_counts: dict[str, int] = {}
    model_costs: dict[str, float] = {}
    for s in sessions:
        model_counts[s.model_name] = model_counts.get(s.model_name, 0) + 1
        model_costs[s.model_name] = model_costs.get(s.model_name, 0) + s.cost_usd

    print(f"\n    Models:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        cost = model_costs[model]
        print(f"      {model}: {count:,} sessions, ${cost:.4f} cost")

    # By date
    date_counts: dict[str, int] = {}
    for s in sessions:
        date_key = f"{s.year}-{s.month}-{s.day}"
        date_counts[date_key] = date_counts.get(date_key, 0) + 1

    print(f"\n    Date range: {min(date_counts.keys())} to {max(date_counts.keys())}")
    print(f"    Days with data: {len(date_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for Talki session logs")
    parser.add_argument("--sessions", type=int, default=500, help="Number of sessions to generate")
    parser.add_argument("--days", type=int, default=30, help="Number of days to spread data across")
    parser.add_argument("--stage", default="prod", help="Stage (prod/dev)")
    parser.add_argument("--bucket", default=S3_BUCKET, help="S3 bucket name")
    parser.add_argument("--prefix", default=S3_PREFIX, help="S3 key prefix")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    parser.add_argument("--local", action="store_true", help="Save to local files instead of S3")

    args = parser.parse_args()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "   📊 Talki Test Data Generator".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    print(f"\n  Configuration:")
    print(f"    Sessions to generate: {args.sessions:,}")
    print(f"    Date range: {start_date.date()} to {end_date.date()}")
    print(f"    Stage: {args.stage}")
    print(f"    Target: {'Local files' if args.local else f's3://{args.bucket}/{args.prefix}/'}")

    # Generate sessions
    print(f"\n  Generating {args.sessions:,} sessions...")
    sessions = generate_sessions(
        num_sessions=args.sessions,
        start_date=start_date,
        end_date=end_date,
        stage=args.stage,
    )

    # Preview and stats
    preview_sessions(sessions)
    print_statistics(sessions)

    if args.local:
        # Save to local files
        output_dir = os.path.join(os.path.dirname(__file__), "../data/generated")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"sessions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, "w") as f:
            for session in sessions:
                f.write(session.to_json() + "\n")

        print(f"\n  ✅ Saved to {output_file}")
    else:
        # Upload to S3
        print(f"\n  {'[DRY RUN] ' if args.dry_run else ''}Uploading to S3...")
        stats = upload_to_s3(sessions, args.bucket, args.prefix, dry_run=args.dry_run)

        if not args.dry_run:
            print(f"\n  ✅ Uploaded {stats['total_records']:,} records in {stats['files_uploaded']} files")
            print(f"\n  Next steps:")
            print(f"    1. Run MSCK REPAIR TABLE session_logs to refresh partitions")
            print(f"    2. Query the new data in Athena")

    return 0


if __name__ == "__main__":
    sys.exit(main())
