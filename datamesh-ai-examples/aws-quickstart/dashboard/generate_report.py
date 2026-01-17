#!/usr/bin/env python3
"""
Talki Metrics Dashboard Generator

Generates an interactive HTML dashboard with charts and metrics
from AWS Athena data.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../datamesh-ai-connectors/aws-athena"))

from athena_client import AthenaClient, AthenaConfig


@dataclass
class MetricCard:
    """A single metric for display."""
    label: str
    value: str
    change: str | None = None
    change_direction: str = "neutral"  # up, down, neutral


@dataclass
class ChartData:
    """Data for a chart."""
    chart_type: str  # bar, line, pie, table
    title: str
    labels: list[str]
    datasets: list[dict]
    options: dict | None = None


class DashboardGenerator:
    """Generates HTML dashboard from Athena data."""

    def __init__(self, database: str = "talki_metrics_prod"):
        self.athena = AthenaClient(AthenaConfig(
            database=database,
            workgroup="primary",
            output_location="s3://talki-athena-results-eu-west-1/datamesh-ai/",
            region="eu-west-1",
        ))
        self.data: dict[str, Any] = {}

    def fetch_all_data(self):
        """Fetch all data for the dashboard."""
        print("Fetching dashboard data from Athena...")

        # Summary metrics
        print("  - Summary metrics...")
        self.data["summary"] = self._fetch_summary()

        # Sessions by language
        print("  - Sessions by language...")
        self.data["by_language"] = self._fetch_by_language()

        # Cost by model
        print("  - Cost by model...")
        self.data["by_model"] = self._fetch_by_model()

        # Daily trend
        print("  - Daily trend...")
        self.data["daily_trend"] = self._fetch_daily_trend()

        # Regional distribution
        print("  - Regional distribution...")
        self.data["by_region"] = self._fetch_by_region()

        # Error analysis
        print("  - Error analysis...")
        self.data["errors"] = self._fetch_errors()

        # Top families
        print("  - Top families...")
        self.data["top_families"] = self._fetch_top_families()

        print("Data fetched successfully!")

    def _fetch_summary(self) -> dict:
        result = self.athena.execute_query("""
            SELECT
                COUNT(*) as total_sessions,
                COUNT(DISTINCT family_id_hash) as unique_families,
                COUNT(DISTINCT child_id_hash) as unique_children,
                COUNT(DISTINCT language) as languages,
                ROUND(SUM(cost_usd), 4) as total_cost,
                ROUND(AVG(latency_ms), 0) as avg_latency_ms,
                ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate,
                SUM(input_tokens + output_tokens) as total_tokens
            FROM session_logs
        """)
        return result.rows[0] if result.rows else {}

    def _fetch_by_language(self) -> list[dict]:
        result = self.athena.execute_query("""
            SELECT
                language,
                COUNT(*) as sessions,
                COUNT(DISTINCT family_id_hash) as families,
                ROUND(SUM(cost_usd), 4) as cost,
                ROUND(AVG(latency_ms), 0) as avg_latency
            FROM session_logs
            GROUP BY language
            ORDER BY sessions DESC
            LIMIT 10
        """)
        return result.rows

    def _fetch_by_model(self) -> list[dict]:
        result = self.athena.execute_query("""
            SELECT
                model_name,
                COUNT(*) as sessions,
                ROUND(SUM(cost_usd), 4) as total_cost,
                ROUND(AVG(cost_usd), 6) as avg_cost,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                ROUND(AVG(latency_ms), 0) as avg_latency
            FROM session_logs
            GROUP BY model_name
            ORDER BY total_cost DESC
        """)
        return result.rows

    def _fetch_daily_trend(self) -> list[dict]:
        result = self.athena.execute_query("""
            SELECT
                CONCAT(year, '-', month, '-', day) as date,
                COUNT(*) as sessions,
                COUNT(DISTINCT family_id_hash) as families,
                ROUND(SUM(cost_usd), 4) as cost,
                ROUND(AVG(latency_ms), 0) as avg_latency,
                ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
            FROM session_logs
            GROUP BY year, month, day
            ORDER BY year, month, day
        """)
        return result.rows

    def _fetch_by_region(self) -> list[dict]:
        result = self.athena.execute_query("""
            SELECT
                region,
                COUNT(*) as sessions,
                ROUND(SUM(cost_usd), 4) as cost,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
            FROM session_logs
            GROUP BY region
            ORDER BY sessions DESC
        """)
        return result.rows

    def _fetch_errors(self) -> list[dict]:
        result = self.athena.execute_query("""
            SELECT
                COALESCE(error_code, 'Success') as status,
                COUNT(*) as count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
            FROM session_logs
            GROUP BY error_code
            ORDER BY count DESC
        """)
        return result.rows

    def _fetch_top_families(self) -> list[dict]:
        result = self.athena.execute_query("""
            SELECT
                family_id_hash,
                COUNT(*) as sessions,
                COUNT(DISTINCT child_id_hash) as children,
                ROUND(SUM(cost_usd), 4) as total_cost,
                ROUND(AVG(latency_ms), 0) as avg_latency
            FROM session_logs
            GROUP BY family_id_hash
            ORDER BY sessions DESC
            LIMIT 10
        """)
        return result.rows

    def generate_html(self) -> str:
        """Generate the HTML dashboard."""
        summary = self.data.get("summary", {})

        # Prepare chart data
        language_labels = [r["language"] for r in self.data.get("by_language", [])]
        language_sessions = [int(r["sessions"]) for r in self.data.get("by_language", [])]
        language_costs = [float(r["cost"]) for r in self.data.get("by_language", [])]

        model_labels = [r["model_name"].split("-")[0] + "..." for r in self.data.get("by_model", [])]
        model_costs = [float(r["total_cost"]) for r in self.data.get("by_model", [])]

        daily_dates = [r["date"] for r in self.data.get("daily_trend", [])]
        daily_sessions = [int(r["sessions"]) for r in self.data.get("daily_trend", [])]
        daily_costs = [float(r["cost"]) for r in self.data.get("daily_trend", [])]

        region_labels = [r["region"] for r in self.data.get("by_region", [])]
        region_values = [int(r["sessions"]) for r in self.data.get("by_region", [])]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Talki Metrics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }}
        .header {{
            background: rgba(255,255,255,0.05);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 24px;
            font-weight: 600;
        }}
        .header p {{
            opacity: 0.7;
            font-size: 14px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .metric-card .label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.6;
            margin-bottom: 8px;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #4ade80;
        }}
        .metric-card .value.cost {{
            color: #f472b6;
        }}
        .metric-card .value.latency {{
            color: #60a5fa;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        @media (max-width: 900px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .chart-card h3 {{
            font-size: 16px;
            margin-bottom: 20px;
            opacity: 0.9;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 1px;
            opacity: 0.6;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge.success {{
            background: rgba(74, 222, 128, 0.2);
            color: #4ade80;
        }}
        .badge.error {{
            background: rgba(248, 113, 113, 0.2);
            color: #f87171;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            opacity: 0.5;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Talki Metrics Dashboard</h1>
        <p>AI-Powered Educational Conversations Analytics | Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
    </div>

    <div class="container">
        <!-- Summary Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Total Sessions</div>
                <div class="value">{int(summary.get('total_sessions', 0)):,}</div>
            </div>
            <div class="metric-card">
                <div class="label">Unique Families</div>
                <div class="value">{int(summary.get('unique_families', 0)):,}</div>
            </div>
            <div class="metric-card">
                <div class="label">Languages</div>
                <div class="value">{int(summary.get('languages', 0))}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Cost</div>
                <div class="value cost">${float(summary.get('total_cost', 0)):,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Latency</div>
                <div class="value latency">{int(float(summary.get('avg_latency_ms', 0))):,}ms</div>
            </div>
            <div class="metric-card">
                <div class="label">Success Rate</div>
                <div class="value">{float(summary.get('success_rate', 0)):.1f}%</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="charts-grid">
            <!-- Sessions by Language -->
            <div class="chart-card">
                <h3>Sessions by Language</h3>
                <div class="chart-container">
                    <canvas id="languageChart"></canvas>
                </div>
            </div>

            <!-- Cost by Model -->
            <div class="chart-card">
                <h3>Cost by Model</h3>
                <div class="chart-container">
                    <canvas id="modelChart"></canvas>
                </div>
            </div>

            <!-- Daily Trend -->
            <div class="chart-card full-width">
                <h3>Daily Sessions & Cost Trend</h3>
                <div class="chart-container">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>

            <!-- Regional Distribution -->
            <div class="chart-card">
                <h3>Regional Distribution</h3>
                <div class="chart-container">
                    <canvas id="regionChart"></canvas>
                </div>
            </div>

            <!-- Model Performance -->
            <div class="chart-card">
                <h3>Model Performance</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Sessions</th>
                            <th>Cost</th>
                            <th>Avg Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''<tr>
                            <td>{r["model_name"][:30]}</td>
                            <td>{int(r["sessions"]):,}</td>
                            <td>${float(r["total_cost"]):.4f}</td>
                            <td>{int(float(r["avg_latency"])):,}ms</td>
                        </tr>''' for r in self.data.get("by_model", []))}
                    </tbody>
                </table>
            </div>

            <!-- Top Families -->
            <div class="chart-card full-width">
                <h3>Top 10 Most Active Families</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Family ID</th>
                            <th>Sessions</th>
                            <th>Children</th>
                            <th>Total Cost</th>
                            <th>Avg Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''<tr>
                            <td><code>{r["family_id_hash"][:12]}...</code></td>
                            <td>{int(r["sessions"]):,}</td>
                            <td>{int(r["children"]):,}</td>
                            <td>${float(r["total_cost"]):.4f}</td>
                            <td>{int(float(r["avg_latency"])):,}ms</td>
                        </tr>''' for r in self.data.get("top_families", []))}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="footer">
        Generated by DataMesh.AI | Powered by AWS Athena
    </div>

    <script>
        // Chart defaults
        Chart.defaults.color = 'rgba(255,255,255,0.7)';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';

        // Language Chart
        new Chart(document.getElementById('languageChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(language_labels)},
                datasets: [{{
                    label: 'Sessions',
                    data: {json.dumps(language_sessions)},
                    backgroundColor: 'rgba(74, 222, 128, 0.5)',
                    borderColor: 'rgba(74, 222, 128, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});

        // Model Cost Chart
        new Chart(document.getElementById('modelChart'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps([r["model_name"] for r in self.data.get("by_model", [])])},
                datasets: [{{
                    data: {json.dumps(model_costs)},
                    backgroundColor: [
                        'rgba(244, 114, 182, 0.7)',
                        'rgba(96, 165, 250, 0.7)',
                        'rgba(74, 222, 128, 0.7)',
                        'rgba(250, 204, 21, 0.7)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});

        // Daily Trend Chart
        new Chart(document.getElementById('trendChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(daily_dates)},
                datasets: [{{
                    label: 'Sessions',
                    data: {json.dumps(daily_sessions)},
                    borderColor: 'rgba(74, 222, 128, 1)',
                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                    fill: true,
                    yAxisID: 'y'
                }}, {{
                    label: 'Cost ($)',
                    data: {json.dumps(daily_costs)},
                    borderColor: 'rgba(244, 114, 182, 1)',
                    backgroundColor: 'rgba(244, 114, 182, 0.1)',
                    fill: false,
                    yAxisID: 'y1'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        type: 'linear',
                        position: 'left',
                        title: {{ display: true, text: 'Sessions' }}
                    }},
                    y1: {{
                        type: 'linear',
                        position: 'right',
                        title: {{ display: true, text: 'Cost ($)' }},
                        grid: {{ drawOnChartArea: false }}
                    }}
                }}
            }}
        }});

        // Regional Chart
        new Chart(document.getElementById('regionChart'), {{
            type: 'pie',
            data: {{
                labels: {json.dumps(region_labels)},
                datasets: [{{
                    data: {json.dumps(region_values)},
                    backgroundColor: [
                        'rgba(74, 222, 128, 0.7)',
                        'rgba(96, 165, 250, 0.7)',
                        'rgba(244, 114, 182, 0.7)',
                        'rgba(250, 204, 21, 0.7)',
                        'rgba(167, 139, 250, 0.7)',
                        'rgba(248, 113, 113, 0.7)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
    </script>
</body>
</html>
"""
        return html

    def save_dashboard(self, output_path: str | None = None):
        """Generate and save the dashboard."""
        self.fetch_all_data()
        html = self.generate_html()

        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), "index.html")

        with open(output_path, "w") as f:
            f.write(html)

        print(f"\nDashboard saved to: {output_path}")
        print(f"Open in browser: file://{os.path.abspath(output_path)}")


def main():
    print("\n" + "=" * 60)
    print("  Talki Metrics Dashboard Generator")
    print("=" * 60)

    generator = DashboardGenerator(
        database=os.environ.get("ATHENA_DATABASE", "talki_metrics_prod")
    )
    generator.save_dashboard()

    return 0


if __name__ == "__main__":
    sys.exit(main())
