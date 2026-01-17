"""
SQL Generator - Natural language to SQL conversion.

Generates SQL queries from natural language questions using
schema context and governance constraints.
"""

import logging
import re
from typing import Any, Optional

from datamesh_ai_core.a2a import A2AClient

logger = logging.getLogger(__name__)


class SQLGenerator:
    """
    Generates SQL from natural language questions.

    Uses schema information and user context to produce
    valid, governed SQL queries.
    """

    def __init__(self, a2a_client: Optional[A2AClient] = None):
        self.a2a_client = a2a_client
        self._templates = self._load_templates()

    def _load_templates(self) -> dict[str, str]:
        """Load SQL generation templates."""
        return {
            "revenue_monthly": """
                SELECT
                    DATE_TRUNC('month', transaction_date) AS month,
                    region,
                    SUM(amount) AS total_revenue
                FROM {table}
                WHERE YEAR(transaction_date) = {year}
                GROUP BY DATE_TRUNC('month', transaction_date), region
                ORDER BY month, region
            """,
            "top_customers": """
                SELECT
                    customer_id,
                    COUNT(*) AS transaction_count,
                    SUM(amount) AS total_amount
                FROM {table}
                GROUP BY customer_id
                ORDER BY total_amount DESC
                LIMIT {limit}
            """,
            "aggregation": """
                SELECT
                    {group_by},
                    {aggregations}
                FROM {table}
                {where_clause}
                GROUP BY {group_by}
                {order_clause}
            """,
        }

    async def generate(
        self,
        question: str,
        schema_info: dict,
        constraints: dict,
        user_context: dict,
    ) -> dict[str, Any]:
        """
        Generate SQL from a natural language question.

        Args:
            question: Natural language question
            schema_info: Schema information from catalog
            constraints: Query constraints (maxRows, etc.)
            user_context: User context for governance

        Returns:
            Generated SQL with metadata
        """
        logger.info(f"Generating SQL for: {question}")

        # Parse the question to understand intent
        intent = self._parse_intent(question)

        # Extract tables from schema info
        tables = self._extract_tables(schema_info)

        # Generate SQL based on intent
        sql = self._generate_sql(intent, tables, constraints)

        # Apply governance filters
        sql = self._apply_governance_filters(sql, user_context)

        # Add row limit if specified
        max_rows = constraints.get("maxRows", 10000)
        if "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip().rstrip(';')}\nLIMIT {max_rows}"

        return {
            "sql": sql,
            "explanation": self._generate_explanation(intent, tables),
            "tables": tables,
            "confidence": self._calculate_confidence(intent),
        }

    def _parse_intent(self, question: str) -> dict:
        """Parse question to understand user intent."""
        question_lower = question.lower()

        intent = {
            "type": "select",
            "aggregations": [],
            "group_by": [],
            "filters": [],
            "time_range": None,
            "limit": None,
            "order_by": None,
        }

        # Detect aggregations
        if any(word in question_lower for word in ["sum", "total", "revenue"]):
            intent["aggregations"].append("SUM")
        if any(word in question_lower for word in ["count", "how many"]):
            intent["aggregations"].append("COUNT")
        if any(word in question_lower for word in ["average", "avg", "mean"]):
            intent["aggregations"].append("AVG")

        # Detect grouping
        if "by region" in question_lower:
            intent["group_by"].append("region")
        if "monthly" in question_lower or "by month" in question_lower:
            intent["group_by"].append("month")
        if "by customer" in question_lower:
            intent["group_by"].append("customer_id")

        # Detect time range
        year_match = re.search(r'\b(20\d{2})\b', question)
        if year_match:
            intent["time_range"] = {"year": int(year_match.group(1))}

        # Detect top N
        top_match = re.search(r'top\s+(\d+)', question_lower)
        if top_match:
            intent["limit"] = int(top_match.group(1))
            intent["order_by"] = "DESC"

        return intent

    def _extract_tables(self, schema_info: dict) -> list[str]:
        """Extract table names from schema info."""
        tables = []
        if "fields" in schema_info:
            # Schema contains field definitions, extract table from context
            tables.append(schema_info.get("table", "finance.revenue"))
        if "tables" in schema_info:
            tables.extend(schema_info["tables"])
        if not tables:
            tables.append("finance.revenue")
        return tables

    def _generate_sql(
        self,
        intent: dict,
        tables: list[str],
        constraints: dict,
    ) -> str:
        """Generate SQL from parsed intent."""
        table = tables[0] if tables else "finance.revenue"

        # Check for revenue monthly pattern
        if "month" in intent["group_by"] and "SUM" in intent["aggregations"]:
            year = intent.get("time_range", {}).get("year", 2025)
            return self._templates["revenue_monthly"].format(
                table=table,
                year=year,
            ).strip()

        # Check for top customers pattern
        if intent.get("limit") and "customer_id" in intent["group_by"]:
            return self._templates["top_customers"].format(
                table=table,
                limit=intent.get("limit", 10),
            ).strip()

        # Generic aggregation query
        group_by = ", ".join(intent["group_by"]) or "1"
        aggregations = self._build_aggregations(intent)
        where_clause = self._build_where_clause(intent)
        order_clause = self._build_order_clause(intent)

        return self._templates["aggregation"].format(
            table=table,
            group_by=group_by,
            aggregations=aggregations,
            where_clause=where_clause,
            order_clause=order_clause,
        ).strip()

    def _build_aggregations(self, intent: dict) -> str:
        """Build aggregation expressions."""
        aggs = []
        if "SUM" in intent["aggregations"]:
            aggs.append("SUM(amount) AS total_amount")
        if "COUNT" in intent["aggregations"]:
            aggs.append("COUNT(*) AS count")
        if "AVG" in intent["aggregations"]:
            aggs.append("AVG(amount) AS avg_amount")

        return ", ".join(aggs) if aggs else "COUNT(*) AS count"

    def _build_where_clause(self, intent: dict) -> str:
        """Build WHERE clause from filters."""
        conditions = []

        if intent.get("time_range"):
            year = intent["time_range"].get("year")
            if year:
                conditions.append(f"YEAR(transaction_date) = {year}")

        if conditions:
            return "WHERE " + " AND ".join(conditions)
        return ""

    def _build_order_clause(self, intent: dict) -> str:
        """Build ORDER BY clause."""
        if intent.get("order_by"):
            direction = intent["order_by"]
            if "SUM" in intent["aggregations"]:
                return f"ORDER BY total_amount {direction}"
            return f"ORDER BY count {direction}"
        return ""

    def _apply_governance_filters(self, sql: str, user_context: dict) -> str:
        """Apply governance filters to SQL."""
        # In production, this would apply row-level security
        # and column masking based on user context
        return sql

    def _generate_explanation(self, intent: dict, tables: list[str]) -> str:
        """Generate human-readable explanation of the query."""
        parts = []

        if intent["aggregations"]:
            agg_str = ", ".join(intent["aggregations"])
            parts.append(f"Aggregating data using {agg_str}")

        if intent["group_by"]:
            group_str = ", ".join(intent["group_by"])
            parts.append(f"grouped by {group_str}")

        if intent.get("time_range"):
            year = intent["time_range"].get("year")
            if year:
                parts.append(f"for year {year}")

        parts.append(f"from table(s): {', '.join(tables)}")

        return " ".join(parts)

    def _calculate_confidence(self, intent: dict) -> float:
        """Calculate confidence score for the generated query."""
        confidence = 0.5  # Base confidence

        # Higher confidence if we detected clear intent
        if intent["aggregations"]:
            confidence += 0.2
        if intent["group_by"]:
            confidence += 0.15
        if intent.get("time_range"):
            confidence += 0.1

        return min(confidence, 0.95)
