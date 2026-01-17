#!/usr/bin/env python3
"""
Natural Language Query Demo

Interactive demonstration of the NL-to-SQL engine with real Athena execution.
"""

from __future__ import annotations

import json
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../datamesh-ai-connectors/aws-athena"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../agents"))

from athena_client import AthenaClient, AthenaConfig
from nl_to_sql import NLToSQLEngine


def print_header(text: str):
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def format_value(val):
    """Format a value for display."""
    if val is None:
        return "-"
    try:
        f = float(val)
        if f > 1000:
            return f"{f:,.0f}"
        elif f > 1:
            return f"{f:.2f}"
        else:
            return f"{f:.4f}"
    except (ValueError, TypeError):
        return str(val)[:40]


def print_results(result, max_rows=15):
    """Print query results as a formatted table."""
    if not result.rows:
        print("  No results returned.")
        return

    # Get column widths
    columns = result.columns
    widths = {col: max(len(col), 8) for col in columns}

    for row in result.rows[:max_rows]:
        for col in columns:
            val_str = format_value(row.get(col))
            widths[col] = max(widths[col], min(len(val_str), 25))

    # Print header
    header = " | ".join(col[:widths[col]].ljust(widths[col]) for col in columns)
    print(f"  {header}")
    print("  " + "-" * len(header))

    # Print rows
    for row in result.rows[:max_rows]:
        row_str = " | ".join(
            format_value(row.get(col))[:widths[col]].ljust(widths[col])
            for col in columns
        )
        print(f"  {row_str}")

    if len(result.rows) > max_rows:
        print(f"  ... and {len(result.rows) - max_rows} more rows")


def run_nl_query(engine: NLToSQLEngine, athena: AthenaClient, question: str):
    """Run a natural language query and display results."""
    print(f"\n📝 Question: {question}")
    print("-" * 70)

    # Generate SQL
    query = engine.generate(question)

    print(f"💡 Confidence: {query.confidence * 100:.0f}%")
    print(f"📊 Suggested visualizations: {', '.join(query.suggested_visualizations)}")

    if query.optimization_hints:
        print(f"⚡ Optimization hints: {', '.join(query.optimization_hints)}")

    print(f"\n📄 Generated SQL:")
    for line in query.sql.split("\n"):
        print(f"    {line}")

    # Execute
    print(f"\n🔄 Executing query...")
    result = athena.execute_query(query.sql, max_rows=100)

    if result.state.value == "SUCCEEDED":
        print(f"✅ Success! ({result.row_count} rows, {result.bytes_scanned:,} bytes scanned, {result.execution_time_ms}ms)")
        print()
        print_results(result)
    else:
        print(f"❌ Query failed: {result.error_message}")


def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "   🗣️  Natural Language Query Demo".center(68) + "║")
    print("║" + "   DataMesh.AI + AWS Athena".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Initialize
    engine = NLToSQLEngine()
    athena = AthenaClient(AthenaConfig(
        database=os.environ.get("ATHENA_DATABASE", "talki_metrics_prod"),
        workgroup="primary",
        output_location="s3://talki-athena-results-eu-west-1/datamesh-ai/",
        region="eu-west-1",
    ))

    # Demo questions
    demo_questions = [
        "How many sessions do we have by language?",
        "What's our total cost by model?",
        "Show me the daily trend of sessions",
        "What's the success rate by model?",
        "Top 5 most active regions",
        "Compare latency across models",
    ]

    print_header("DEMO QUERIES")
    print("  Running a series of natural language queries against Talki metrics...")

    for question in demo_questions:
        run_nl_query(engine, athena, question)
        print()

    print_header("EXECUTIVE SUMMARY")

    # Run summary queries
    summary_result = athena.execute_query("""
        SELECT
            COUNT(*) as total_sessions,
            COUNT(DISTINCT family_id_hash) as unique_families,
            COUNT(DISTINCT language) as languages,
            ROUND(SUM(cost_usd), 2) as total_cost,
            ROUND(AVG(latency_ms), 0) as avg_latency_ms,
            ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
        FROM session_logs
    """)

    if summary_result.rows:
        row = summary_result.rows[0]
        print(f"""
  📊 Talki Metrics Summary
  ────────────────────────────────────────────────────────────
    Total Sessions:     {row['total_sessions']:>10}
    Unique Families:    {row['unique_families']:>10}
    Languages:          {row['languages']:>10}
    Total Cost:         ${float(row['total_cost']):>9,.2f}
    Avg Latency:        {row['avg_latency_ms']:>10}ms
    Success Rate:       {row['success_rate']:>9}%
  ────────────────────────────────────────────────────────────
        """)

    print_header("INTERACTIVE MODE")
    print("  Enter a natural language question (or 'exit' to quit):")

    while True:
        try:
            question = input("\n  You: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("  Goodbye!")
                break
            run_nl_query(engine, athena, question)
        except KeyboardInterrupt:
            print("\n  Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
