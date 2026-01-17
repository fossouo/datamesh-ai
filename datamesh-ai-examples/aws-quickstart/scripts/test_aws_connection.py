#!/usr/bin/env python3
"""
AWS Connection Test Script

Tests connectivity to AWS Glue Catalog and Athena
before running the full DataMesh.AI demo.
"""

import json
import os
import sys

# Add connectors to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../datamesh-ai-connectors/aws-athena"))

def print_header(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text: str):
    print(f"  [OK] {text}")


def print_error(text: str):
    print(f"  [ERROR] {text}")


def print_info(text: str):
    print(f"  [INFO] {text}")


def test_aws_credentials():
    """Test AWS credentials are configured."""
    print_header("Testing AWS Credentials")

    try:
        import boto3
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        print_success(f"AWS Account: {identity['Account']}")
        print_success(f"User ARN: {identity['Arn']}")
        return True
    except Exception as e:
        print_error(f"AWS credentials not configured: {e}")
        print_info("Run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False


def test_glue_catalog():
    """Test Glue Catalog access."""
    print_header("Testing Glue Catalog Access")

    try:
        from glue_catalog import GlueCatalogClient, GlueCatalogConfig

        config = GlueCatalogConfig(
            region=os.environ.get("AWS_REGION", "eu-west-1"),
            database_filter="talki_.*",
        )
        client = GlueCatalogClient(config)

        # List databases
        databases = client.list_databases()
        print_success(f"Found {len(databases)} databases")
        for db in databases[:5]:
            print_info(f"  - {db}")
        if len(databases) > 5:
            print_info(f"  ... and {len(databases) - 5} more")

        # Check for Talki metrics database
        target_db = os.environ.get("ATHENA_DATABASE", "talki_metrics_dev")
        if target_db in databases:
            print_success(f"Target database exists: {target_db}")

            # List tables
            tables = client.list_tables(target_db)
            print_success(f"Found {len(tables)} tables in {target_db}")
            for table in tables[:10]:
                print_info(f"  - {table}")

            # Get schema for session_logs if it exists
            if "session_logs" in tables:
                table_info = client.get_table(target_db, "session_logs")
                if table_info:
                    print_success(f"session_logs schema retrieved")
                    print_info(f"  Columns: {len(table_info.columns)}")
                    print_info(f"  Partition keys: {len(table_info.partition_keys)}")
                    print_info(f"  Location: {table_info.location}")
        else:
            print_error(f"Target database not found: {target_db}")
            print_info("Available databases:")
            for db in databases:
                print_info(f"  - {db}")

        return True

    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Run: pip install boto3")
        return False
    except Exception as e:
        print_error(f"Glue Catalog error: {e}")
        return False


def test_athena_query():
    """Test Athena query execution."""
    print_header("Testing Athena Query Execution")

    try:
        from athena_client import AthenaClient, AthenaConfig, QueryState

        database = os.environ.get("ATHENA_DATABASE", "talki_metrics_dev")
        output_location = os.environ.get(
            "ATHENA_OUTPUT",
            "s3://talki-athena-results-eu-west-1/datamesh-ai/"
        )

        config = AthenaConfig(
            database=database,
            workgroup="primary",
            output_location=output_location,
            region=os.environ.get("AWS_REGION", "eu-west-1"),
        )
        client = AthenaClient(config)

        # Run a simple test query
        print_info(f"Database: {database}")
        print_info(f"Output: {output_location}")
        print_info("Running test query...")

        result = client.execute_query(
            "SELECT COUNT(*) as total FROM session_logs LIMIT 1",
            max_rows=10,
            timeout_seconds=60,
        )

        if result.state == QueryState.SUCCEEDED:
            print_success(f"Query executed successfully")
            print_info(f"  Execution ID: {result.query_execution_id}")
            print_info(f"  Bytes scanned: {result.bytes_scanned:,}")
            print_info(f"  Execution time: {result.execution_time_ms}ms")
            if result.rows:
                total = result.rows[0].get("total", "N/A")
                print_success(f"  Total sessions in database: {total}")
            return True
        else:
            print_error(f"Query failed: {result.error_message}")
            return False

    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        return False
    except Exception as e:
        print_error(f"Athena error: {e}")
        return False


def test_sample_queries():
    """Run sample Talki analytics queries."""
    print_header("Running Sample Analytics Queries")

    try:
        from athena_client import AthenaClient, AthenaConfig, QueryState

        database = os.environ.get("ATHENA_DATABASE", "talki_metrics_dev")
        output_location = os.environ.get(
            "ATHENA_OUTPUT",
            "s3://talki-athena-results-eu-west-1/datamesh-ai/"
        )

        config = AthenaConfig(
            database=database,
            output_location=output_location,
            region=os.environ.get("AWS_REGION", "eu-west-1"),
        )
        client = AthenaClient(config)

        queries = [
            ("Sessions by Language", """
                SELECT language, COUNT(*) as sessions
                FROM session_logs
                GROUP BY language
                ORDER BY sessions DESC
                LIMIT 5
            """),
            ("Cost by Model", """
                SELECT model_name, ROUND(SUM(cost_usd), 4) as total_cost
                FROM session_logs
                GROUP BY model_name
                ORDER BY total_cost DESC
                LIMIT 5
            """),
            ("Daily Activity (Last 7 Days)", """
                SELECT year, month, day, COUNT(*) as sessions
                FROM session_logs
                GROUP BY year, month, day
                ORDER BY year DESC, month DESC, day DESC
                LIMIT 7
            """),
        ]

        for name, query in queries:
            print(f"\n  Query: {name}")
            result = client.execute_query(query.strip(), max_rows=10, timeout_seconds=60)

            if result.state == QueryState.SUCCEEDED:
                print_success(f"  Returned {result.row_count} rows")
                for row in result.rows[:3]:
                    print_info(f"    {json.dumps(row)}")
                if result.row_count > 3:
                    print_info(f"    ... and {result.row_count - 3} more")
            else:
                print_error(f"  Failed: {result.error_message}")

        return True

    except Exception as e:
        print_error(f"Sample queries error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  DATAMESH.AI - AWS Connection Test")
    print("=" * 60)

    results = {
        "AWS Credentials": test_aws_credentials(),
    }

    # Only continue if credentials work
    if results["AWS Credentials"]:
        results["Glue Catalog"] = test_glue_catalog()
        results["Athena Query"] = test_athena_query()

        if results["Athena Query"]:
            results["Sample Queries"] = test_sample_queries()

    # Summary
    print_header("Test Summary")
    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed! You can now run the AWS quickstart demo.")
        print("  Run: make run-demo")
    else:
        print("  Some tests failed. Please check the errors above.")
        print("  Common issues:")
        print("    1. AWS credentials not configured")
        print("    2. Missing IAM permissions for Glue/Athena")
        print("    3. Database or tables don't exist yet")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
