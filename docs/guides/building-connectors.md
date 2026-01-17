# Building Connectors for DataMesh-AI

This guide explains how to create a new connector to integrate DataMesh-AI with your data source, catalog, or processing engine.

## Overview

Connectors are the bridge between DataMesh-AI agents and external systems. They provide a standardized interface that agents use to:

- **Query Engines**: Execute SQL, get schemas, explain plans
- **Data Catalogs**: Discover tables, get lineage, check classifications
- **Processing Engines**: Submit jobs, check status, get results
- **Storage Systems**: Read/write data, list files

## Connector Types

| Type | Purpose | Examples |
|------|---------|----------|
| `QueryConnector` | Execute queries against SQL engines | Trino, Spark SQL, BigQuery |
| `CatalogConnector` | Access data catalog metadata | Unity, Glue, Atlas |
| `ProcessingConnector` | Submit and manage jobs | Spark, dbt, Airflow |
| `StorageConnector` | Access raw data storage | S3, GCS, ADLS |

---

## Quick Start

### 1. Create Connector Structure

```
packages/connectors/
└── your-connector/
    ├── __init__.py
    ├── connector.py          # Main implementation
    ├── config.py             # Configuration schema
    ├── exceptions.py         # Custom exceptions
    ├── tests/
    │   ├── __init__.py
    │   ├── test_connector.py
    │   └── conftest.py
    ├── README.md
    └── pyproject.toml
```

### 2. Implement the Interface

```python
# connector.py
from datamesh_ai_core.connectors import QueryConnector, ConnectionConfig
from datamesh_ai_core.models import QueryResult, Schema, Table

class YourConnector(QueryConnector):
    """Connector for YourDatabase."""

    name = "your-connector"
    version = "0.1.0"

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Establish connection to the data source."""
        self._client = YourClient(
            host=self.config.host,
            port=self.config.port,
            # ...
        )
        await self._client.connect()

    async def disconnect(self) -> None:
        """Close the connection."""
        if self._client:
            await self._client.close()

    async def execute(self, query: str) -> QueryResult:
        """Execute a SQL query and return results."""
        result = await self._client.execute(query)
        return QueryResult(
            columns=result.columns,
            rows=result.rows,
            row_count=len(result.rows),
            execution_time_ms=result.execution_time,
        )

    async def get_schema(self, table: str) -> Schema:
        """Get the schema for a table."""
        # Implementation
        pass

    async def list_tables(self, schema: str = None) -> list[Table]:
        """List available tables."""
        # Implementation
        pass

    async def explain(self, query: str) -> str:
        """Get the execution plan for a query."""
        # Implementation
        pass
```

### 3. Define Configuration

```python
# config.py
from pydantic import BaseModel, Field
from typing import Optional

class YourConnectorConfig(BaseModel):
    """Configuration for YourConnector."""

    host: str = Field(..., description="Database host")
    port: int = Field(default=8080, description="Database port")
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password (use env var)")
    catalog: Optional[str] = Field(default=None, description="Default catalog")
    schema_name: Optional[str] = Field(default=None, alias="schema", description="Default schema")

    # Connection pool settings
    pool_size: int = Field(default=5, ge=1, le=50)
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    class Config:
        extra = "forbid"  # Reject unknown fields
```

---

## Connector Interface Reference

### QueryConnector

For SQL query engines:

```python
class QueryConnector(ABC):
    """Interface for SQL query engines."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""

    @abstractmethod
    async def execute(self, query: str) -> QueryResult:
        """Execute a query."""

    @abstractmethod
    async def get_schema(self, table: str) -> Schema:
        """Get table schema."""

    @abstractmethod
    async def list_tables(self, schema: str = None) -> list[Table]:
        """List tables."""

    @abstractmethod
    async def explain(self, query: str) -> str:
        """Get query plan."""

    # Optional methods with default implementations
    async def health_check(self) -> bool:
        """Check if connection is healthy."""
        try:
            await self.execute("SELECT 1")
            return True
        except Exception:
            return False
```

### CatalogConnector

For data catalogs:

```python
class CatalogConnector(ABC):
    """Interface for data catalogs."""

    @abstractmethod
    async def get_table_metadata(self, table: str) -> TableMetadata:
        """Get full metadata for a table."""

    @abstractmethod
    async def search_tables(self, query: str) -> list[TableSummary]:
        """Search for tables by name or description."""

    @abstractmethod
    async def get_lineage(self, table: str) -> Lineage:
        """Get upstream/downstream lineage."""

    @abstractmethod
    async def get_classifications(self, table: str) -> list[Classification]:
        """Get data classifications (PII, etc.)."""

    @abstractmethod
    async def list_schemas(self, catalog: str = None) -> list[str]:
        """List available schemas."""
```

### ProcessingConnector

For job execution engines:

```python
class ProcessingConnector(ABC):
    """Interface for processing engines."""

    @abstractmethod
    async def submit_job(self, job: JobSpec) -> JobHandle:
        """Submit a job for execution."""

    @abstractmethod
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get current job status."""

    @abstractmethod
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get job results (when complete)."""

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""

    @abstractmethod
    async def list_jobs(self, status: str = None) -> list[JobSummary]:
        """List jobs."""
```

---

## Best Practices

### 1. Error Handling

```python
from datamesh_ai_core.exceptions import (
    ConnectorError,
    ConnectionError,
    QueryError,
    AuthenticationError,
)

async def execute(self, query: str) -> QueryResult:
    try:
        result = await self._client.execute(query)
        return self._transform_result(result)
    except YourClientAuthError as e:
        raise AuthenticationError(f"Authentication failed: {e}")
    except YourClientTimeoutError as e:
        raise QueryError(f"Query timeout: {e}", query=query)
    except Exception as e:
        raise ConnectorError(f"Unexpected error: {e}")
```

### 2. Connection Pooling

```python
class YourConnector(QueryConnector):
    def __init__(self, config: YourConnectorConfig):
        self.config = config
        self._pool = None

    async def connect(self) -> None:
        self._pool = await create_pool(
            host=self.config.host,
            port=self.config.port,
            min_size=1,
            max_size=self.config.pool_size,
        )

    async def execute(self, query: str) -> QueryResult:
        async with self._pool.acquire() as conn:
            return await conn.execute(query)
```

### 3. Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class YourConnector(QueryConnector):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def execute(self, query: str) -> QueryResult:
        # Retries on transient failures
        ...
```

### 4. Observability

```python
from datamesh_ai_core.tracing import trace, SpanKind

class YourConnector(QueryConnector):
    @trace(name="your_connector.execute", kind=SpanKind.CLIENT)
    async def execute(self, query: str) -> QueryResult:
        # Automatically traced
        ...
```

---

## Testing Your Connector

### Unit Tests

```python
# tests/test_connector.py
import pytest
from unittest.mock import AsyncMock, patch

from your_connector import YourConnector, YourConnectorConfig

@pytest.fixture
def config():
    return YourConnectorConfig(
        host="localhost",
        port=8080,
    )

@pytest.fixture
def connector(config):
    return YourConnector(config)

@pytest.mark.asyncio
async def test_execute_simple_query(connector):
    with patch.object(connector, '_client') as mock_client:
        mock_client.execute = AsyncMock(return_value=MockResult())

        result = await connector.execute("SELECT 1")

        assert result.row_count == 1
        mock_client.execute.assert_called_once_with("SELECT 1")

@pytest.mark.asyncio
async def test_connection_error(connector):
    with patch.object(connector, '_client') as mock_client:
        mock_client.connect = AsyncMock(side_effect=ConnectionRefused())

        with pytest.raises(ConnectionError):
            await connector.connect()
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
import os

@pytest.fixture
def real_connector():
    """Uses real connection - requires TEST_DB_HOST env var."""
    if not os.getenv("TEST_DB_HOST"):
        pytest.skip("Integration tests require TEST_DB_HOST")

    return YourConnector(YourConnectorConfig(
        host=os.getenv("TEST_DB_HOST"),
        port=int(os.getenv("TEST_DB_PORT", "8080")),
    ))

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_query(real_connector):
    await real_connector.connect()
    try:
        result = await real_connector.execute("SELECT 1 as test")
        assert result.row_count == 1
    finally:
        await real_connector.disconnect()
```

---

## Registering Your Connector

### 1. Add to Registry

```python
# packages/connectors/__init__.py
from datamesh_ai_core.registry import register_connector
from .your_connector import YourConnector

register_connector("your-connector", YourConnector)
```

### 2. Add CLI Support

```yaml
# packages/cli/connectors.yaml
connectors:
  your-connector:
    class: your_connector.YourConnector
    config_class: your_connector.YourConnectorConfig
    description: "Connect to YourDatabase"
    docs_url: "https://docs.datamesh-ai.io/connectors/your-connector"
```

### 3. Document It

Create `packages/connectors/your-connector/README.md`:

```markdown
# YourConnector for DataMesh-AI

Connect DataMesh-AI to YourDatabase.

## Installation

\`\`\`bash
pip install datamesh-connector-your-db
\`\`\`

## Configuration

\`\`\`yaml
connectors:
  - type: your-connector
    host: your-db.example.com
    port: 8080
    username: ${YOUR_DB_USER}
    password: ${YOUR_DB_PASSWORD}
\`\`\`

## Usage

\`\`\`bash
dmesh connector add your-connector --host your-db.example.com
\`\`\`

## Features

- [x] Query execution
- [x] Schema introspection
- [x] Query explain
- [ ] Write support (coming soon)
```

---

## Community Connectors

Community-contributed connectors go in `community/connectors/`. They follow the same structure but are maintained by the community.

To submit a community connector:

1. Fork the repository
2. Create your connector in `community/connectors/your-connector/`
3. Include tests and documentation
4. Submit a PR

---

## Need Help?

- **Discord**: Ask in #connector-dev channel
- **GitHub Discussions**: Open a Q&A discussion
- **Examples**: See `packages/connectors/trino/` for reference

Happy building! 🚀
