"""
Natural Language to SQL Engine for DataMesh.AI

Schema-aware NL-to-SQL generation using pattern matching, semantic understanding,
and intelligent query building. Designed for the Talki metrics domain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class SchemaField:
    """A field in a table schema."""
    name: str
    data_type: str
    description: str = ""
    is_partition: bool = False
    is_metric: bool = False
    is_dimension: bool = False
    examples: list[str] = field(default_factory=list)


@dataclass
class TableSchema:
    """Schema information for a table."""
    name: str
    database: str
    description: str
    fields: list[SchemaField]
    partition_keys: list[str] = field(default_factory=list)

    @property
    def dimensions(self) -> list[SchemaField]:
        return [f for f in self.fields if f.is_dimension]

    @property
    def metrics(self) -> list[SchemaField]:
        return [f for f in self.fields if f.is_metric]


@dataclass
class SQLQuery:
    """A generated SQL query with metadata."""
    sql: str
    explanation: str
    confidence: float
    tables_used: list[str]
    suggested_visualizations: list[str] = field(default_factory=list)
    optimization_hints: list[str] = field(default_factory=list)


# Talki session_logs schema with rich metadata
TALKI_SESSION_LOGS = TableSchema(
    name="session_logs",
    database="talki_metrics_prod",
    description="Session-level metrics for AI-powered educational conversations",
    fields=[
        SchemaField("session_id", "string", "Unique session identifier", is_dimension=True),
        SchemaField("family_id_hash", "string", "Hashed family identifier for privacy", is_dimension=True),
        SchemaField("child_id_hash", "string", "Hashed child identifier for privacy", is_dimension=True),
        SchemaField("language", "string", "Session language (e.g., en-GB, es-ES)", is_dimension=True,
                   examples=["en-GB", "en-US", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "nl-NL", "pl-PL", "ja-JP"]),
        SchemaField("region", "string", "AWS region where session was processed", is_dimension=True,
                   examples=["eu-west-1", "eu-west-2", "us-east-1", "us-west-2", "ap-northeast-1", "sa-east-1"]),
        SchemaField("source", "string", "Client source (mobile_app, web_app, etc.)", is_dimension=True,
                   examples=["mobile_app", "web_app", "alexa_skill", "google_assistant"]),
        SchemaField("stage", "string", "Deployment stage (prod, dev)", is_dimension=True),
        SchemaField("timestamp", "string", "ISO timestamp of session"),
        SchemaField("duration_seconds", "int", "Session duration in seconds", is_metric=True),
        SchemaField("latency_ms", "int", "AI response latency in milliseconds", is_metric=True),
        SchemaField("cost_usd", "double", "Total cost in USD for API calls", is_metric=True),
        SchemaField("success", "boolean", "Whether session completed successfully", is_dimension=True),
        SchemaField("error_code", "string", "Error code if session failed", is_dimension=True,
                   examples=["TIMEOUT", "RATE_LIMIT", "CONTENT_FILTER", "MODEL_ERROR"]),
        SchemaField("model_provider", "string", "AI model provider (anthropic, openai)", is_dimension=True,
                   examples=["anthropic", "openai"]),
        SchemaField("model_name", "string", "Specific model used", is_dimension=True,
                   examples=["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229", "gpt-4o-mini"]),
        SchemaField("input_tokens", "int", "Number of input tokens", is_metric=True),
        SchemaField("output_tokens", "int", "Number of output tokens", is_metric=True),
        SchemaField("year", "string", "Year partition (YYYY)", is_partition=True),
        SchemaField("month", "string", "Month partition (MM)", is_partition=True),
        SchemaField("day", "string", "Day partition (DD)", is_partition=True),
    ],
    partition_keys=["year", "month", "day"],
)


class NLToSQLEngine:
    """
    Natural Language to SQL conversion engine.

    Uses pattern matching, semantic understanding, and schema awareness
    to generate optimized SQL queries from natural language questions.
    """

    def __init__(self, schemas: list[TableSchema] | None = None):
        self.schemas = schemas or [TALKI_SESSION_LOGS]
        self.schema_map = {s.name: s for s in self.schemas}

    def generate(self, question: str, context: dict[str, Any] | None = None) -> SQLQuery:
        """
        Generate SQL from a natural language question.

        Args:
            question: Natural language question
            context: Optional context (user info, time range, etc.)

        Returns:
            SQLQuery with generated SQL and metadata
        """
        question_lower = question.lower()
        context = context or {}

        # Detect intent
        intent = self._detect_intent(question_lower)

        # Build query based on intent
        if intent == "count":
            return self._build_count_query(question_lower, context)
        elif intent == "cost":
            return self._build_cost_query(question_lower, context)
        elif intent == "performance":
            return self._build_performance_query(question_lower, context)
        elif intent == "trend":
            return self._build_trend_query(question_lower, context)
        elif intent == "breakdown":
            return self._build_breakdown_query(question_lower, context)
        elif intent == "comparison":
            return self._build_comparison_query(question_lower, context)
        elif intent == "top_n":
            return self._build_top_n_query(question_lower, context)
        elif intent == "error":
            return self._build_error_query(question_lower, context)
        else:
            return self._build_overview_query(question_lower, context)

    def _detect_intent(self, question: str) -> str:
        """Detect the intent/type of question."""

        # Count questions
        if any(kw in question for kw in ["how many", "count", "total sessions", "number of"]):
            return "count"

        # Cost questions
        if any(kw in question for kw in ["cost", "spend", "expense", "price", "billing", "budget"]):
            return "cost"

        # Performance questions
        if any(kw in question for kw in ["latency", "performance", "slow", "fast", "response time", "duration"]):
            return "performance"

        # Trend questions
        if any(kw in question for kw in ["trend", "daily", "weekly", "monthly", "over time", "timeline", "history"]):
            return "trend"

        # Breakdown/distribution questions
        if any(kw in question for kw in ["by language", "by region", "by model", "breakdown", "distribution", "per"]):
            return "breakdown"

        # Comparison questions
        if any(kw in question for kw in ["compare", "versus", "vs", "difference between"]):
            return "comparison"

        # Top N questions
        if any(kw in question for kw in ["top", "best", "worst", "most", "least", "highest", "lowest"]):
            return "top_n"

        # Error questions
        if any(kw in question for kw in ["error", "fail", "issue", "problem", "success rate"]):
            return "error"

        return "overview"

    def _extract_dimensions(self, question: str) -> list[str]:
        """Extract dimension columns mentioned in the question."""
        dimensions = []

        dimension_keywords = {
            "language": "language",
            "region": "region",
            "model": "model_name",
            "provider": "model_provider",
            "source": "source",
            "family": "family_id_hash",
            "child": "child_id_hash",
            "country": "region",
            "app": "source",
        }

        for keyword, column in dimension_keywords.items():
            if keyword in question and column not in dimensions:
                dimensions.append(column)

        return dimensions

    def _extract_time_filter(self, question: str, context: dict) -> tuple[str, str]:
        """Extract time filters from question or context."""
        today = datetime.utcnow()

        # Check for explicit time ranges
        if "today" in question:
            return today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "yesterday" in question:
            yesterday = today - timedelta(days=1)
            return yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")
        elif "last week" in question:
            start = today - timedelta(days=7)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "last month" in question or "past month" in question:
            start = today - timedelta(days=30)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "last 7 days" in question:
            start = today - timedelta(days=7)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "last 30 days" in question:
            start = today - timedelta(days=30)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        # Default: all data (no filter)
        return "", ""

    def _build_partition_filter(self, start_date: str, end_date: str) -> str:
        """Build partition filter for date range."""
        if not start_date:
            return ""

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else start

        # Build efficient partition filter
        if start.year == end.year and start.month == end.month:
            # Same month
            if start.day == end.day:
                return f"year = '{start.year}' AND month = '{start.month:02d}' AND day = '{start.day:02d}'"
            else:
                return f"year = '{start.year}' AND month = '{start.month:02d}' AND day BETWEEN '{start.day:02d}' AND '{end.day:02d}'"
        else:
            # Cross-month range - use timestamp filter (less efficient but necessary)
            return f"timestamp >= '{start_date}' AND timestamp < '{(end + timedelta(days=1)).strftime('%Y-%m-%d')}'"

    def _build_count_query(self, question: str, context: dict) -> SQLQuery:
        """Build a count/sessions query."""
        dimensions = self._extract_dimensions(question)
        start_date, end_date = self._extract_time_filter(question, context)

        if dimensions:
            group_by = ", ".join(dimensions)
            select_dims = ", ".join(dimensions)
            sql = f"""SELECT
    {select_dims},
    COUNT(*) as sessions,
    COUNT(DISTINCT family_id_hash) as unique_families
FROM session_logs"""
            if start_date:
                sql += f"\nWHERE {self._build_partition_filter(start_date, end_date)}"
            sql += f"""
GROUP BY {group_by}
ORDER BY sessions DESC
LIMIT 50"""
        else:
            sql = """SELECT
    COUNT(*) as total_sessions,
    COUNT(DISTINCT family_id_hash) as unique_families,
    COUNT(DISTINCT child_id_hash) as unique_children
FROM session_logs"""
            if start_date:
                sql += f"\nWHERE {self._build_partition_filter(start_date, end_date)}"

        return SQLQuery(
            sql=sql,
            explanation=f"Count query: {question}",
            confidence=0.9,
            tables_used=["session_logs"],
            suggested_visualizations=["bar_chart", "metric_card"] if dimensions else ["metric_card"],
        )

    def _build_cost_query(self, question: str, context: dict) -> SQLQuery:
        """Build a cost analysis query."""
        dimensions = self._extract_dimensions(question)
        start_date, end_date = self._extract_time_filter(question, context)

        if not dimensions:
            # Check for common cost breakdowns
            if "model" in question:
                dimensions = ["model_name"]
            elif "language" in question:
                dimensions = ["language"]
            elif "region" in question:
                dimensions = ["region"]
            else:
                dimensions = ["model_name"]  # Default to model breakdown

        group_by = ", ".join(dimensions)
        select_dims = ", ".join(dimensions)

        sql = f"""SELECT
    {select_dims},
    COUNT(*) as sessions,
    ROUND(SUM(cost_usd), 4) as total_cost,
    ROUND(AVG(cost_usd), 6) as avg_cost_per_session,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    ROUND(SUM(cost_usd) / NULLIF(SUM(input_tokens + output_tokens), 0) * 1000, 6) as cost_per_1k_tokens
FROM session_logs"""

        where_clauses = []
        if start_date:
            where_clauses.append(self._build_partition_filter(start_date, end_date))

        if where_clauses:
            sql += f"\nWHERE {' AND '.join(where_clauses)}"

        sql += f"""
GROUP BY {group_by}
ORDER BY total_cost DESC"""

        return SQLQuery(
            sql=sql,
            explanation=f"Cost analysis by {', '.join(dimensions)}",
            confidence=0.92,
            tables_used=["session_logs"],
            suggested_visualizations=["bar_chart", "pie_chart", "table"],
            optimization_hints=["Consider adding date filters to reduce scan costs"],
        )

    def _build_performance_query(self, question: str, context: dict) -> SQLQuery:
        """Build a performance/latency query."""
        dimensions = self._extract_dimensions(question)
        start_date, end_date = self._extract_time_filter(question, context)

        if not dimensions:
            if "model" in question:
                dimensions = ["model_name"]
            else:
                dimensions = ["model_name", "language"]

        group_by = ", ".join(dimensions)
        select_dims = ", ".join(dimensions)

        sql = f"""SELECT
    {select_dims},
    COUNT(*) as sessions,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms,
    MIN(latency_ms) as min_latency_ms,
    MAX(latency_ms) as max_latency_ms,
    ROUND(APPROX_PERCENTILE(latency_ms, 0.95), 2) as p95_latency_ms,
    ROUND(AVG(duration_seconds), 2) as avg_duration_sec
FROM session_logs"""

        where_clauses = ["latency_ms > 0"]
        if start_date:
            where_clauses.append(self._build_partition_filter(start_date, end_date))

        sql += f"\nWHERE {' AND '.join(where_clauses)}"
        sql += f"""
GROUP BY {group_by}
ORDER BY avg_latency_ms DESC"""

        return SQLQuery(
            sql=sql,
            explanation=f"Performance analysis by {', '.join(dimensions)}",
            confidence=0.88,
            tables_used=["session_logs"],
            suggested_visualizations=["bar_chart", "line_chart", "heatmap"],
        )

    def _build_trend_query(self, question: str, context: dict) -> SQLQuery:
        """Build a time-based trend query."""
        dimensions = self._extract_dimensions(question)
        start_date, end_date = self._extract_time_filter(question, context)

        # Determine time granularity
        if "hourly" in question:
            time_expr = "CONCAT(year, '-', month, '-', day, ' ', SUBSTR(timestamp, 12, 2), ':00')"
            time_alias = "hour"
        elif "weekly" in question:
            time_expr = "date_trunc('week', date_parse(timestamp, '%Y-%m-%dT%H:%i:%sZ'))"
            time_alias = "week"
        else:  # Default to daily
            time_expr = "CONCAT(year, '-', month, '-', day)"
            time_alias = "date"

        # Build select
        select_parts = [f"{time_expr} as {time_alias}"]
        if dimensions:
            select_parts.extend(dimensions)

        sql = f"""SELECT
    {', '.join(select_parts)},
    COUNT(*) as sessions,
    COUNT(DISTINCT family_id_hash) as active_families,
    ROUND(SUM(cost_usd), 4) as daily_cost,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms,
    ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
FROM session_logs"""

        where_clauses = []
        if start_date:
            where_clauses.append(self._build_partition_filter(start_date, end_date))

        if where_clauses:
            sql += f"\nWHERE {' AND '.join(where_clauses)}"

        # Use the expression for GROUP BY (Athena doesn't support alias in GROUP BY)
        group_exprs = [time_expr] + dimensions
        sql += f"""
GROUP BY {', '.join(group_exprs)}
ORDER BY {time_alias} DESC
LIMIT 90"""

        return SQLQuery(
            sql=sql,
            explanation=f"Time trend analysis",
            confidence=0.85,
            tables_used=["session_logs"],
            suggested_visualizations=["line_chart", "area_chart"],
            optimization_hints=["Use partition filters for large date ranges"],
        )

    def _build_breakdown_query(self, question: str, context: dict) -> SQLQuery:
        """Build a breakdown/distribution query."""
        dimensions = self._extract_dimensions(question)
        start_date, end_date = self._extract_time_filter(question, context)

        if not dimensions:
            # Try to detect from question
            if "language" in question:
                dimensions = ["language"]
            elif "region" in question:
                dimensions = ["region"]
            elif "model" in question:
                dimensions = ["model_name"]
            elif "source" in question or "app" in question:
                dimensions = ["source"]
            else:
                dimensions = ["language", "region"]

        group_by = ", ".join(dimensions)
        select_dims = ", ".join(dimensions)

        sql = f"""SELECT
    {select_dims},
    COUNT(*) as sessions,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage,
    COUNT(DISTINCT family_id_hash) as unique_families,
    ROUND(SUM(cost_usd), 4) as total_cost,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms
FROM session_logs"""

        where_clauses = []
        if start_date:
            where_clauses.append(self._build_partition_filter(start_date, end_date))

        if where_clauses:
            sql += f"\nWHERE {' AND '.join(where_clauses)}"

        sql += f"""
GROUP BY {group_by}
ORDER BY sessions DESC"""

        return SQLQuery(
            sql=sql,
            explanation=f"Breakdown by {', '.join(dimensions)}",
            confidence=0.9,
            tables_used=["session_logs"],
            suggested_visualizations=["pie_chart", "bar_chart", "treemap"],
        )

    def _build_comparison_query(self, question: str, context: dict) -> SQLQuery:
        """Build a comparison query."""
        # Extract what's being compared
        dimensions = self._extract_dimensions(question)

        if not dimensions:
            dimensions = ["model_name"]  # Default comparison

        group_by = ", ".join(dimensions)

        sql = f"""SELECT
    {group_by},
    COUNT(*) as sessions,
    ROUND(SUM(cost_usd), 4) as total_cost,
    ROUND(AVG(cost_usd), 6) as avg_cost,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms,
    ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate,
    SUM(input_tokens + output_tokens) as total_tokens
FROM session_logs
GROUP BY {group_by}
ORDER BY sessions DESC"""

        return SQLQuery(
            sql=sql,
            explanation=f"Comparison by {', '.join(dimensions)}",
            confidence=0.85,
            tables_used=["session_logs"],
            suggested_visualizations=["grouped_bar_chart", "radar_chart", "table"],
        )

    def _build_top_n_query(self, question: str, context: dict) -> SQLQuery:
        """Build a top N query."""
        # Determine N
        n = 10
        match = re.search(r"top\s+(\d+)", question)
        if match:
            n = int(match.group(1))

        # Determine what to rank by
        if "cost" in question or "expensive" in question:
            order_col = "total_cost"
            order_dir = "DESC"
        elif "latency" in question or "slow" in question:
            order_col = "avg_latency_ms"
            order_dir = "DESC"
        elif "fast" in question:
            order_col = "avg_latency_ms"
            order_dir = "ASC"
        elif "active" in question or "users" in question:
            order_col = "unique_families"
            order_dir = "DESC"
        else:
            order_col = "sessions"
            order_dir = "DESC"

        # Determine grouping
        dimensions = self._extract_dimensions(question)
        if not dimensions:
            if "language" in question:
                dimensions = ["language"]
            elif "family" in question or "user" in question:
                dimensions = ["family_id_hash"]
            else:
                dimensions = ["model_name"]

        group_by = ", ".join(dimensions)

        sql = f"""SELECT
    {group_by},
    COUNT(*) as sessions,
    COUNT(DISTINCT family_id_hash) as unique_families,
    ROUND(SUM(cost_usd), 4) as total_cost,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms
FROM session_logs
GROUP BY {group_by}
ORDER BY {order_col} {order_dir}
LIMIT {n}"""

        return SQLQuery(
            sql=sql,
            explanation=f"Top {n} by {order_col}",
            confidence=0.88,
            tables_used=["session_logs"],
            suggested_visualizations=["bar_chart", "table"],
        )

    def _build_error_query(self, question: str, context: dict) -> SQLQuery:
        """Build an error analysis query."""
        start_date, end_date = self._extract_time_filter(question, context)

        if "success rate" in question or "rate" in question:
            sql = """SELECT
    model_name,
    COUNT(*) as total_sessions,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
    ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate
FROM session_logs
GROUP BY model_name
ORDER BY success_rate ASC"""
        else:
            sql = """SELECT
    error_code,
    model_name,
    COUNT(*) as occurrences,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
FROM session_logs
WHERE NOT success AND error_code IS NOT NULL AND error_code != ''
GROUP BY error_code, model_name
ORDER BY occurrences DESC
LIMIT 20"""

        return SQLQuery(
            sql=sql,
            explanation="Error and success rate analysis",
            confidence=0.9,
            tables_used=["session_logs"],
            suggested_visualizations=["bar_chart", "pie_chart", "table"],
        )

    def _build_overview_query(self, question: str, context: dict) -> SQLQuery:
        """Build a general overview query."""
        sql = """SELECT
    language,
    region,
    COUNT(*) as sessions,
    COUNT(DISTINCT family_id_hash) as families,
    ROUND(SUM(cost_usd), 4) as total_cost,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms,
    ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
FROM session_logs
GROUP BY language, region
ORDER BY sessions DESC
LIMIT 20"""

        return SQLQuery(
            sql=sql,
            explanation="General overview of session metrics",
            confidence=0.75,
            tables_used=["session_logs"],
            suggested_visualizations=["table", "heatmap"],
            optimization_hints=["Add specific filters or groupings for more targeted insights"],
        )


def demo():
    """Demonstrate the NL-to-SQL engine."""
    engine = NLToSQLEngine()

    questions = [
        "How many sessions do we have by language?",
        "What's our cost by model?",
        "Show me the performance breakdown by model",
        "Daily trend of sessions for the last 30 days",
        "Top 5 most expensive languages",
        "What's our error rate by model?",
        "Compare costs between models",
        "Show me session distribution by region",
    ]

    print("\n" + "═" * 70)
    print("  NL-to-SQL Engine Demo")
    print("═" * 70)

    for q in questions:
        result = engine.generate(q)
        print(f"\n📝 Question: {q}")
        print(f"💡 Confidence: {result.confidence * 100:.0f}%")
        print(f"📊 Visualizations: {', '.join(result.suggested_visualizations)}")
        print(f"📄 SQL:\n{result.sql[:300]}...")
        print("-" * 70)


if __name__ == "__main__":
    demo()
