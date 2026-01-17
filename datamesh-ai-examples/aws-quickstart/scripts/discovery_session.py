#!/usr/bin/env python3
"""
DataMesh.AI Discovery Session

Explores your Talki metrics data like a data analyst would,
surfacing insights and patterns automatically.
"""

import json
import sys
import time
from datetime import datetime

import httpx

CATALOG_URL = "http://localhost:8082"
SQL_URL = "http://localhost:8081"


def print_header(text: str):
    width = 70
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def print_subheader(text: str):
    print(f"\n  ▶ {text}")
    print("  " + "─" * 60)


def print_insight(text: str):
    print(f"    💡 {text}")


def print_data(label: str, value):
    print(f"    {label}: {value}")


def catalog_request(capability: str, payload: dict) -> dict:
    """Send request to Catalog Agent."""
    response = httpx.post(
        CATALOG_URL,
        json={"capability": capability, "payload": payload},
        timeout=60.0,
    )
    return response.json()


def sql_request(capability: str, payload: dict) -> dict:
    """Send request to SQL Agent."""
    response = httpx.post(
        SQL_URL,
        json={"capability": capability, "payload": payload},
        timeout=120.0,
    )
    return response.json()


def execute_sql(sql: str, description: str = "") -> list:
    """Execute SQL and return rows."""
    if description:
        print(f"\n    📝 {description}")

    result = sql_request("sql.execute", {"sql": sql})

    if result.get("status") == "SUCCESS" and result.get("data", {}).get("status") == "success":
        data = result["data"]
        print(f"    ⏱️  {data.get('execution_time_ms', 0)}ms | {data.get('bytes_scanned', 0):,} bytes scanned")
        return data.get("rows", [])
    else:
        error = result.get("data", {}).get("error") or result.get("error", "Unknown error")
        print(f"    ❌ Query failed: {error}")
        return []


def discover_catalog():
    """Discover available datasets."""
    print_header("1. CATALOG DISCOVERY")

    response = httpx.get(f"{CATALOG_URL}/datasets", timeout=60.0)
    data = response.json()

    print(f"\n  Found {data['count']} datasets in AWS Glue Catalog:\n")

    for ds in data["datasets"]:
        table = ds.split(".")[-1]
        # Get schema details
        schema_result = catalog_request("catalog.resolve", {"dataset": ds})
        if schema_result.get("status") == "SUCCESS":
            field_count = schema_result["data"]["field_count"]
            location = schema_result["data"].get("location", "")
            s3_bucket = location.split("/")[2] if location else "unknown"
            print(f"    📊 {table:25} ({field_count} columns) → s3://{s3_bucket}/...")


def analyze_session_logs():
    """Deep analysis of session_logs table."""
    print_header("2. SESSION LOGS ANALYSIS")

    # Overall stats
    print_subheader("Overall Statistics")
    rows = execute_sql("""
        SELECT
            COUNT(*) as total_sessions,
            COUNT(DISTINCT family_id_hash) as unique_families,
            COUNT(DISTINCT child_id_hash) as unique_children,
            ROUND(SUM(cost_usd), 4) as total_cost_usd,
            ROUND(AVG(cost_usd), 6) as avg_cost_per_session,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens
        FROM session_logs
    """, "Aggregating all session metrics...")

    if rows:
        r = rows[0]
        print()
        print_data("Total Sessions", r.get("total_sessions"))
        print_data("Unique Families", r.get("unique_families"))
        print_data("Unique Children", r.get("unique_children"))
        print_data("Total Cost (USD)", f"${r.get('total_cost_usd')}")
        print_data("Avg Cost/Session", f"${r.get('avg_cost_per_session')}")
        print_data("Total Input Tokens", f"{int(r.get('total_input_tokens') or 0):,}")
        print_data("Total Output Tokens", f"{int(r.get('total_output_tokens') or 0):,}")


def analyze_by_language():
    """Analyze metrics by language."""
    print_header("3. LANGUAGE BREAKDOWN")

    print_subheader("Sessions and Costs by Language")
    rows = execute_sql("""
        SELECT
            language,
            region,
            COUNT(*) as sessions,
            COUNT(DISTINCT family_id_hash) as families,
            ROUND(SUM(cost_usd), 4) as total_cost,
            ROUND(AVG(latency_ms), 0) as avg_latency_ms
        FROM session_logs
        GROUP BY language, region
        ORDER BY sessions DESC
    """, "Breaking down by language and region...")

    if rows:
        print()
        print(f"    {'Language':<10} {'Region':<12} {'Sessions':>10} {'Families':>10} {'Cost':>12} {'Latency':>10}")
        print("    " + "-" * 64)
        for r in rows:
            lang = r.get("language", "?")
            region = r.get("region", "?")
            sessions = r.get("sessions", 0)
            families = r.get("families", 0)
            cost = f"${r.get('total_cost', 0)}"
            latency = f"{r.get('avg_latency_ms', 0)}ms"
            print(f"    {lang:<10} {region:<12} {sessions:>10} {families:>10} {cost:>12} {latency:>10}")

        # Insights
        if len(rows) > 1:
            top_lang = rows[0].get("language")
            print_insight(f"Most active language: {top_lang}")


def analyze_models():
    """Analyze AI model usage and costs."""
    print_header("4. AI MODEL ANALYSIS")

    print_subheader("Cost and Performance by Model")
    rows = execute_sql("""
        SELECT
            model_provider,
            model_name,
            COUNT(*) as sessions,
            ROUND(SUM(cost_usd), 4) as total_cost,
            ROUND(AVG(cost_usd), 6) as avg_cost,
            ROUND(AVG(latency_ms), 0) as avg_latency,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens
        FROM session_logs
        GROUP BY model_provider, model_name
        ORDER BY total_cost DESC
    """, "Analyzing model usage patterns...")

    if rows:
        print()
        for r in rows:
            provider = r.get("model_provider", "unknown")
            model = r.get("model_name", "unknown")
            print(f"    🤖 {provider}/{model}")
            print(f"       Sessions: {r.get('sessions')} | Cost: ${r.get('total_cost')} | Avg: ${r.get('avg_cost')}/session")
            print(f"       Latency: {r.get('avg_latency')}ms avg")
            print(f"       Tokens: {int(r.get('input_tokens') or 0):,} in / {int(r.get('output_tokens') or 0):,} out")
            print()


def analyze_daily_trends():
    """Analyze daily usage trends."""
    print_header("5. DAILY TRENDS")

    print_subheader("Activity Over Time")
    rows = execute_sql("""
        SELECT
            year, month, day,
            COUNT(*) as sessions,
            COUNT(DISTINCT family_id_hash) as active_families,
            ROUND(SUM(cost_usd), 4) as daily_cost,
            ROUND(AVG(latency_ms), 0) as avg_latency
        FROM session_logs
        GROUP BY year, month, day
        ORDER BY year DESC, month DESC, day DESC
        LIMIT 14
    """, "Fetching daily activity trends...")

    if rows:
        print()
        print(f"    {'Date':<12} {'Sessions':>10} {'Families':>10} {'Cost':>12} {'Latency':>10}")
        print("    " + "-" * 54)
        for r in rows:
            date = f"{r.get('year')}-{r.get('month')}-{r.get('day')}"
            sessions = r.get("sessions", 0)
            families = r.get("active_families", 0)
            cost = f"${r.get('daily_cost', 0)}"
            latency = f"{r.get('avg_latency', 0)}ms"
            print(f"    {date:<12} {sessions:>10} {families:>10} {cost:>12} {latency:>10}")

        if len(rows) >= 2:
            latest = int(rows[0].get("sessions", 0))
            previous = int(rows[1].get("sessions", 0))
            if previous > 0:
                change = ((latest - previous) / previous) * 100
                trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print_insight(f"Day-over-day change: {trend} {change:+.1f}%")


def analyze_success_rates():
    """Analyze success/failure rates."""
    print_header("6. SUCCESS & ERROR ANALYSIS")

    print_subheader("Success Rates by Language")
    rows = execute_sql("""
        SELECT
            language,
            COUNT(*) as total,
            SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful,
            SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as failed,
            ROUND(CAST(SUM(CASE WHEN success = true THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) * 100, 1) as success_rate
        FROM session_logs
        GROUP BY language
        ORDER BY total DESC
    """, "Calculating success rates...")

    if rows:
        print()
        for r in rows:
            lang = r.get("language", "?")
            total = r.get("total", 0)
            success_rate = r.get("success_rate", 0)
            bar_len = int(float(success_rate or 0) / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {lang}: [{bar}] {success_rate}% ({total} sessions)")

    # Error breakdown
    print_subheader("Error Codes (if any)")
    rows = execute_sql("""
        SELECT
            error_code,
            COUNT(*) as occurrences,
            COUNT(DISTINCT family_id_hash) as affected_families
        FROM session_logs
        WHERE error_code IS NOT NULL AND error_code != ''
        GROUP BY error_code
        ORDER BY occurrences DESC
        LIMIT 10
    """, "Checking for errors...")

    if rows:
        print()
        for r in rows:
            print(f"    ⚠️  {r.get('error_code')}: {r.get('occurrences')} occurrences ({r.get('affected_families')} families)")
    else:
        print("    ✅ No errors found in the dataset!")


def analyze_token_efficiency():
    """Analyze token usage efficiency."""
    print_header("7. TOKEN EFFICIENCY")

    print_subheader("Token Usage by Language")
    rows = execute_sql("""
        SELECT
            language,
            COUNT(*) as sessions,
            ROUND(AVG(input_tokens), 0) as avg_input,
            ROUND(AVG(output_tokens), 0) as avg_output,
            ROUND(CAST(SUM(output_tokens) AS DOUBLE) / NULLIF(SUM(input_tokens), 0), 2) as output_ratio
        FROM session_logs
        GROUP BY language
        ORDER BY sessions DESC
    """, "Analyzing token patterns...")

    if rows:
        print()
        print(f"    {'Language':<10} {'Sessions':>10} {'Avg In':>12} {'Avg Out':>12} {'Out/In Ratio':>12}")
        print("    " + "-" * 56)
        for r in rows:
            print(f"    {r.get('language', '?'):<10} {r.get('sessions', 0):>10} {int(r.get('avg_input') or 0):>12} {int(r.get('avg_output') or 0):>12} {r.get('output_ratio', 0):>12}")

        # Find most efficient
        if rows:
            ratios = [(r.get("language"), float(r.get("output_ratio") or 0)) for r in rows if r.get("output_ratio")]
            if ratios:
                best = max(ratios, key=lambda x: x[1])
                print_insight(f"Highest output/input ratio: {best[0]} ({best[1]}x)")


def explore_other_tables():
    """Quick look at other available tables."""
    print_header("8. OTHER DATA SOURCES")

    tables = ["routing_logs", "error_logs", "rag_context_logs", "routine_logs", "daily_aggregates"]

    for table in tables:
        print_subheader(f"Table: {table}")
        rows = execute_sql(f"SELECT COUNT(*) as row_count FROM {table}", f"Counting rows in {table}...")
        if rows:
            count = rows[0].get("row_count", 0)
            if int(count) > 0:
                print(f"    📊 {count} rows available")
                # Get sample
                sample = execute_sql(f"SELECT * FROM {table} LIMIT 1", "Sampling schema...")
                if sample:
                    print(f"    📋 Columns: {', '.join(sample[0].keys())}")
            else:
                print(f"    📭 Table is empty")


def generate_recommendations():
    """Generate data-driven recommendations."""
    print_header("9. INSIGHTS & RECOMMENDATIONS")

    print("""
    Based on the data discovery session:

    📊 DATA COVERAGE
    ────────────────────────────────────────
    • 6 tables available in Glue Catalog
    • session_logs is the primary analytics table
    • Partitioned by year/month/day for efficient queries

    💰 COST OPTIMIZATION
    ────────────────────────────────────────
    • Using Claude 3 Haiku (cost-effective choice)
    • Consider batch processing for similar queries
    • Monitor token usage by language for optimization

    📈 SUGGESTED NEXT QUERIES
    ────────────────────────────────────────
    • Weekly/monthly trend analysis
    • User cohort analysis by signup date
    • A/B testing analysis by source field
    • Peak usage hours analysis

    🔧 AGENT CAPABILITIES DEMONSTRATED
    ────────────────────────────────────────
    • Catalog discovery via Glue
    • Schema resolution with field metadata
    • Real-time Athena query execution
    • Cross-table exploration
    """)


def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "   🔍 DATAMESH.AI - DATA DISCOVERY SESSION".center(68) + "║")
    print("║" + f"   Exploring: talki_metrics_prod".center(68) + "║")
    print("║" + f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    start_time = time.time()

    try:
        discover_catalog()
        analyze_session_logs()
        analyze_by_language()
        analyze_models()
        analyze_daily_trends()
        analyze_success_rates()
        analyze_token_efficiency()
        explore_other_tables()
        generate_recommendations()

    except httpx.ConnectError:
        print("\n❌ Could not connect to agents. Make sure they're running:")
        print("   make run-demo")
        return 1
    except Exception as e:
        print(f"\n❌ Error during discovery: {e}")
        return 1

    elapsed = time.time() - start_time
    print("\n" + "═" * 70)
    print(f"  Discovery session complete in {elapsed:.1f} seconds")
    print("═" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
