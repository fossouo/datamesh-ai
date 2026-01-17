# DataMesh.AI AWS Quickstart

Connect DataMesh.AI agents to your real AWS data lake using Athena and Glue Catalog.

## Overview

This quickstart demonstrates:
- **Catalog Agent** → AWS Glue Data Catalog for schema discovery
- **SQL Agent** → AWS Athena for query execution against S3 data

## Prerequisites

1. **AWS CLI configured** with valid credentials
   ```bash
   aws configure
   # Or set environment variables:
   # export AWS_ACCESS_KEY_ID=xxx
   # export AWS_SECRET_ACCESS_KEY=xxx
   ```

2. **IAM Permissions** - Your user/role needs:
   - `glue:GetDatabases`, `glue:GetTables`, `glue:GetTable`
   - `athena:StartQueryExecution`, `athena:GetQueryExecution`, `athena:GetQueryResults`
   - `s3:GetObject`, `s3:PutObject` on your metrics and results buckets

3. **Existing Talki Infrastructure**:
   - Glue Database: `talki_metrics_dev`
   - S3 Bucket: `talki-metrics-dev`
   - Athena Results: `talki-athena-results-eu-west-1`

## Quick Start

```bash
cd datamesh-ai/datamesh-ai-examples/aws-quickstart

# 1. Setup environment
make setup

# 2. Test AWS connectivity
make test-connection

# 3. Run the demo
make run-demo
```

## Configuration

Set these environment variables to customize:

```bash
export AWS_REGION=eu-west-1
export ATHENA_DATABASE=talki_metrics_dev
export ATHENA_OUTPUT=s3://talki-athena-results-eu-west-1/datamesh-ai/
```

## Usage Examples

### List Available Datasets

```bash
curl -s http://localhost:8082/datasets | jq '.datasets'
```

### Resolve Table Schema

```bash
curl -s -X POST http://localhost:8082 \
  -H "Content-Type: application/json" \
  -d '{
    "capability": "catalog.resolve",
    "payload": {"dataset": "catalog://talki_metrics_dev.session_logs"}
  }' | jq '.data.fields'
```

### Generate SQL from Question

```bash
curl -s -X POST http://localhost:8081 \
  -H "Content-Type: application/json" \
  -d '{
    "capability": "sql.generate",
    "payload": {"question": "Show session counts by language"}
  }' | jq '.data.sql'
```

### Execute SQL Query

```bash
curl -s -X POST http://localhost:8081 \
  -H "Content-Type: application/json" \
  -d '{
    "capability": "sql.execute",
    "payload": {
      "sql": "SELECT language, COUNT(*) as sessions FROM session_logs GROUP BY language ORDER BY sessions DESC LIMIT 10"
    }
  }' | jq '.data.rows'
```

### Validate SQL Syntax

```bash
curl -s -X POST http://localhost:8081 \
  -H "Content-Type: application/json" \
  -d '{
    "capability": "sql.validate",
    "payload": {"sql": "SELECT * FROM session_logs LIMIT 1"}
  }' | jq
```

## Available Capabilities

### Catalog Agent (port 8082)

| Capability | Description |
|------------|-------------|
| `catalog.resolve` | Get schema for a dataset URI |
| `catalog.lineage` | Get data lineage information |
| `catalog.search` | Search for datasets by name |
| `catalog.list` | List all available datasets |

### SQL Agent (port 8081)

| Capability | Description |
|------------|-------------|
| `sql.generate` | Generate SQL from natural language |
| `sql.validate` | Validate SQL syntax via EXPLAIN |
| `sql.execute` | Execute SQL and return results |
| `sql.optimize` | Get query optimization suggestions |

## Security

The SQL Agent enforces **read-only access**:
- Only `SELECT`, `WITH`, `SHOW`, `DESCRIBE`, `EXPLAIN` allowed
- `INSERT`, `UPDATE`, `DELETE`, `DROP`, etc. are blocked
- Query execution timeout: 5 minutes
- Max rows returned: 1000

## Troubleshooting

### "Access Denied" errors
- Check IAM permissions for Glue and Athena
- Verify S3 bucket policies allow your role

### "Database not found"
- Ensure `talki_metrics_dev` exists in Glue Catalog
- Check `ATHENA_DATABASE` environment variable

### "Query timeout"
- Add partition filters (year, month, day) to reduce scan
- Check Athena workgroup quotas

### No data returned
- Verify data exists in S3 bucket
- Check Glue table location points to correct S3 path
- Run `MSCK REPAIR TABLE session_logs` to discover new partitions

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Your Request   │────▶│   Catalog Agent  │────▶ AWS Glue Catalog
│                 │     │   (port 8082)    │      (schema metadata)
└─────────────────┘     └──────────────────┘
        │
        │               ┌──────────────────┐
        └──────────────▶│    SQL Agent     │────▶ AWS Athena
                        │   (port 8081)    │      (query S3 data)
                        └──────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │   S3 Bucket      │
                        │ talki-metrics-*  │
                        │ (session_logs)   │
                        └──────────────────┘
```

## Sample Queries for Talki Data

```sql
-- Session overview
SELECT COUNT(*) as total_sessions,
       COUNT(DISTINCT family_id_hash) as unique_families,
       ROUND(SUM(cost_usd), 4) as total_cost
FROM session_logs;

-- Cost by language
SELECT language,
       COUNT(*) as sessions,
       ROUND(SUM(cost_usd), 4) as total_cost
FROM session_logs
GROUP BY language
ORDER BY total_cost DESC;

-- Daily trend
SELECT year, month, day,
       COUNT(*) as sessions,
       ROUND(SUM(cost_usd), 4) as daily_cost
FROM session_logs
GROUP BY year, month, day
ORDER BY year DESC, month DESC, day DESC
LIMIT 30;

-- Performance by model
SELECT model_name,
       COUNT(*) as sessions,
       ROUND(AVG(latency_ms), 2) as avg_latency_ms
FROM session_logs
WHERE latency_ms > 0
GROUP BY model_name;
```
