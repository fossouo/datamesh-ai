"""
SQL Explainer - Query explanation and execution plan analysis.

Provides human-readable explanations of SQL queries and
their expected execution behavior.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SQLExplainer:
    """
    Explains SQL queries and their execution plans.

    Provides insights into query behavior, potential issues,
    and performance characteristics.
    """

    def __init__(self):
        self._known_patterns = self._load_patterns()

    def _load_patterns(self) -> dict[str, dict]:
        """Load known SQL patterns for explanation."""
        return {
            "aggregation": {
                "pattern": r"(SUM|COUNT|AVG|MIN|MAX)\s*\(",
                "description": "Aggregation query that computes summary statistics",
            },
            "group_by": {
                "pattern": r"GROUP\s+BY",
                "description": "Groups rows by specified columns for aggregation",
            },
            "join": {
                "pattern": r"\bJOIN\b",
                "description": "Combines data from multiple tables",
            },
            "subquery": {
                "pattern": r"\(\s*SELECT",
                "description": "Contains a subquery (query within a query)",
            },
            "window": {
                "pattern": r"\bOVER\s*\(",
                "description": "Uses window functions for analytical calculations",
            },
            "cte": {
                "pattern": r"\bWITH\b.*\bAS\s*\(",
                "description": "Uses Common Table Expression (CTE) for readability",
            },
        }

    async def explain(self, sql: str) -> dict[str, Any]:
        """
        Generate explanation for a SQL query.

        Args:
            sql: SQL query to explain

        Returns:
            Explanation, execution plan, and warnings
        """
        logger.info("Generating SQL explanation")

        # Parse query structure
        structure = self._parse_structure(sql)

        # Generate explanation
        explanation = self._generate_explanation(sql, structure)

        # Generate mock execution plan
        execution_plan = self._generate_execution_plan(sql, structure)

        # Estimate cost
        estimated_cost = self._estimate_cost(structure)

        # Check for potential issues
        warnings = self._check_warnings(sql, structure)

        return {
            "explanation": explanation,
            "execution_plan": execution_plan,
            "estimated_cost": estimated_cost,
            "warnings": warnings,
        }

    def _parse_structure(self, sql: str) -> dict:
        """Parse SQL structure for analysis."""
        sql_upper = sql.upper()

        structure = {
            "type": "SELECT",
            "tables": self._extract_tables(sql),
            "columns": self._extract_columns(sql),
            "has_aggregation": bool(re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", sql_upper)),
            "has_group_by": "GROUP BY" in sql_upper,
            "has_join": "JOIN" in sql_upper,
            "has_subquery": bool(re.search(r"\(\s*SELECT", sql_upper)),
            "has_where": "WHERE" in sql_upper,
            "has_order_by": "ORDER BY" in sql_upper,
            "has_limit": "LIMIT" in sql_upper,
            "patterns": [],
        }

        # Identify patterns
        for name, pattern_info in self._known_patterns.items():
            if re.search(pattern_info["pattern"], sql, re.IGNORECASE):
                structure["patterns"].append(name)

        return structure

    def _extract_tables(self, sql: str) -> list[str]:
        """Extract table names from SQL."""
        tables = []

        # FROM clause
        from_match = re.search(r"\bFROM\s+(\w+(?:\.\w+)?)", sql, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))

        # JOIN clauses
        join_matches = re.findall(r"\bJOIN\s+(\w+(?:\.\w+)?)", sql, re.IGNORECASE)
        tables.extend(join_matches)

        return tables

    def _extract_columns(self, sql: str) -> list[str]:
        """Extract column references from SQL."""
        # Simplified column extraction
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            columns_str = select_match.group(1)
            if columns_str.strip() == "*":
                return ["*"]
            # Split by comma, handling functions
            columns = [c.strip().split()[-1] for c in columns_str.split(",")]
            return columns
        return []

    def _generate_explanation(self, sql: str, structure: dict) -> str:
        """Generate human-readable explanation."""
        parts = []

        parts.append("This query")

        if structure["has_aggregation"]:
            parts.append("performs aggregation")
            if structure["has_group_by"]:
                parts.append("grouped by specific columns")

        if structure["has_join"]:
            parts.append(f"joins {len(structure['tables'])} tables")
        elif structure["tables"]:
            parts.append(f"reads from {', '.join(structure['tables'])}")

        if structure["has_where"]:
            parts.append("with filtering conditions")

        if structure["has_order_by"]:
            parts.append("with sorted results")

        if structure["has_limit"]:
            parts.append("limited to a subset of rows")

        # Add pattern descriptions
        for pattern in structure["patterns"]:
            if pattern in self._known_patterns:
                parts.append(f"({self._known_patterns[pattern]['description']})")

        return " ".join(parts) + "."

    def _generate_execution_plan(self, sql: str, structure: dict) -> list[dict]:
        """Generate mock execution plan."""
        steps = []

        # Table scan or index scan
        for table in structure["tables"]:
            scan_type = "Index Scan" if structure["has_where"] else "Full Table Scan"
            steps.append({
                "step": len(steps) + 1,
                "operation": scan_type,
                "object": table,
                "estimatedRows": 10000,
                "estimatedCost": 100.0,
            })

        # Join operation
        if structure["has_join"]:
            steps.append({
                "step": len(steps) + 1,
                "operation": "Hash Join",
                "object": "joined_result",
                "estimatedRows": 50000,
                "estimatedCost": 250.0,
            })

        # Aggregation
        if structure["has_aggregation"]:
            steps.append({
                "step": len(steps) + 1,
                "operation": "Hash Aggregate",
                "object": "aggregated_result",
                "estimatedRows": 100,
                "estimatedCost": 50.0,
            })

        # Sort
        if structure["has_order_by"]:
            steps.append({
                "step": len(steps) + 1,
                "operation": "Sort",
                "object": "sorted_result",
                "estimatedRows": 100,
                "estimatedCost": 25.0,
            })

        # Limit
        if structure["has_limit"]:
            steps.append({
                "step": len(steps) + 1,
                "operation": "Limit",
                "object": "final_result",
                "estimatedRows": 10,
                "estimatedCost": 1.0,
            })

        return steps

    def _estimate_cost(self, structure: dict) -> dict:
        """Estimate query execution cost."""
        base_cost = 10.0

        if structure["has_join"]:
            base_cost *= 2.5
        if structure["has_aggregation"]:
            base_cost *= 1.5
        if structure["has_subquery"]:
            base_cost *= 2.0
        if not structure["has_where"]:
            base_cost *= 3.0
        if structure["has_limit"]:
            base_cost *= 0.5

        return {
            "relativeCost": round(base_cost, 2),
            "estimatedRowsScanned": 10000 if not structure["has_where"] else 1000,
            "complexity": self._determine_complexity(structure),
        }

    def _determine_complexity(self, structure: dict) -> str:
        """Determine query complexity level."""
        score = 0

        if structure["has_join"]:
            score += 2
        if structure["has_subquery"]:
            score += 2
        if structure["has_aggregation"]:
            score += 1
        if not structure["has_where"]:
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"

    def _check_warnings(self, sql: str, structure: dict) -> list[dict]:
        """Check for potential issues and generate warnings."""
        warnings = []

        if not structure["has_where"] and not structure["has_limit"]:
            warnings.append({
                "level": "warning",
                "code": "UNBOUNDED_QUERY",
                "message": "Query has no WHERE or LIMIT clause, may return large result set",
            })

        if "*" in structure["columns"]:
            warnings.append({
                "level": "info",
                "code": "SELECT_STAR",
                "message": "Using SELECT * may return unnecessary columns",
            })

        if structure["has_subquery"]:
            warnings.append({
                "level": "info",
                "code": "SUBQUERY_DETECTED",
                "message": "Consider using JOIN or CTE for better readability and performance",
            })

        # Check for function on column in WHERE
        if re.search(r"WHERE.*\b(YEAR|MONTH|LOWER|UPPER)\s*\(\s*\w+\s*\)", sql, re.IGNORECASE):
            warnings.append({
                "level": "warning",
                "code": "FUNCTION_ON_COLUMN",
                "message": "Functions on columns in WHERE clause may prevent index usage",
            })

        return warnings
