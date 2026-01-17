# Contributing to DataMesh-AI

First off, thank you for considering contributing to DataMesh-AI! It's people like you that make DataMesh-AI such a great tool for the data community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

---

## Code of Conduct

This project and everyone participating in it is governed by the [DataMesh-AI Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## Getting Started

### Types of Contributions

There are many ways to contribute to DataMesh-AI:

| Contribution Type | Description | Good First Issue? |
|-------------------|-------------|-------------------|
| **Bug Reports** | Report issues you've found | ✅ |
| **Documentation** | Improve guides, fix typos | ✅ |
| **Examples** | Share usage examples | ✅ |
| **Bug Fixes** | Fix reported issues | ✅ |
| **New Connectors** | Add data source support | 🔧 |
| **New Agents** | Build specialized agents | 🔧 |
| **Core Features** | Enhance core functionality | 🏗️ |
| **Specifications** | Improve Agent Contract/A2A | 🏗️ |

### Where to Start

1. **Read the documentation** — Understand the architecture and specifications
2. **Browse issues** — Look for `good first issue` or `help wanted` labels
3. **Join Discord** — Introduce yourself and ask questions
4. **Pick something small** — Start with documentation or a simple bug fix

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/datamesh-ai/datamesh-ai/issues) to avoid duplicates.

When you create a bug report, include:

```markdown
## Bug Description
A clear description of the bug.

## Steps to Reproduce
1. Configure connector with '...'
2. Run query '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.11.0]
- DataMesh-AI version: [e.g., 0.1.0]
- Connector: [e.g., Trino 0.1.0]

## Additional Context
Any other relevant information.
```

### Suggesting Features

Feature suggestions are welcome! Please:

1. **Check existing requests** — Your idea might already be discussed
2. **Describe the use case** — What problem does it solve?
3. **Propose a solution** — How do you envision it working?
4. **Consider alternatives** — Are there other ways to achieve this?

### Building Connectors

Connectors are the most impactful contributions! To build a new connector:

1. **Read the Connector Guide** — See `docs/guides/building-connectors.md`
2. **Follow the interface** — Implement `ConnectorInterface`
3. **Add tests** — Include unit and integration tests
4. **Document** — Provide setup instructions and examples

```
packages/connectors/
├── base/                    # Base connector interface
├── trino/                   # Reference implementation
├── spark/
├── bigquery/
└── your-connector/          # Your contribution!
    ├── __init__.py
    ├── connector.py         # Main implementation
    ├── config.py            # Configuration schema
    ├── tests/
    │   ├── test_connector.py
    │   └── conftest.py
    └── README.md            # Setup instructions
```

### Creating Agents

To create a new agent:

1. **Define the contract** — Write `agent.yaml` following the spec
2. **Implement the handler** — Process requests and return results
3. **Define capabilities** — What can your agent do?
4. **Add governance** — Specify data access and restrictions
5. **Test A2A integration** — Ensure it works with other agents

```yaml
# agent.yaml template
apiVersion: datamesh.ai/v1
kind: Agent

metadata:
  name: your-agent
  version: 1.0.0
  description: What your agent does

spec:
  capabilities:
    - id: your-agent.action
      description: What this action does
      inputSchemaRef: schemas/action.input.json
      outputSchemaRef: schemas/action.output.json

  # ... governance, a2a, observability
```

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) Docker for integration tests

### Local Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/datamesh-ai.git
cd datamesh-ai

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run tests to verify setup
pytest
```

### Running Specific Components

```bash
# Run core tests only
pytest packages/core/tests/

# Run a specific agent
pytest packages/agents/sql-agent/tests/

# Run with coverage
pytest --cov=datamesh_ai --cov-report=html

# Run linters
ruff check .
mypy packages/
```

### Docker Development

```bash
# Build development image
docker build -t datamesh-ai-dev -f Dockerfile.dev .

# Run tests in container
docker run datamesh-ai-dev pytest

# Start local stack with dependencies
docker-compose -f docker-compose.dev.yml up
```

---

## Pull Request Process

### Before Submitting

1. **Create an issue first** — Discuss significant changes before coding
2. **Branch from main** — Use a descriptive branch name
3. **Follow the style guide** — Run linters and formatters
4. **Write tests** — Maintain or improve coverage
5. **Update documentation** — Reflect your changes

### Branch Naming

```
feat/short-description    # New features
fix/issue-number          # Bug fixes
docs/what-you-changed     # Documentation
refactor/what-changed     # Code refactoring
test/what-you-tested      # Test additions
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(sql-agent): add query optimization capability
fix(trino): handle connection timeout gracefully
docs(readme): update installation instructions
test(governance): add PII masking tests
refactor(core): simplify registry interface
```

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Related Issues
Fixes #123

## How Has This Been Tested?
Describe your testing approach.

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented hard-to-understand areas
- [ ] I have updated documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests
- [ ] All tests pass locally
```

### Review Process

1. **Automated checks** — CI must pass
2. **Code review** — At least one maintainer approval
3. **Documentation review** — If applicable
4. **Final approval** — Merge by maintainer

---

## Style Guidelines

### Python

We use:
- **Ruff** for linting and formatting
- **mypy** for type checking
- **Black** style formatting (via Ruff)

```python
# Good
def process_query(
    query: str,
    context: QueryContext,
    *,
    timeout_ms: int = 30000,
) -> QueryResult:
    """Process a SQL query with governance checks.

    Args:
        query: The SQL query to process.
        context: Execution context with user info.
        timeout_ms: Query timeout in milliseconds.

    Returns:
        The query result with metadata.

    Raises:
        PermissionDeniedError: If user lacks access.
        QueryTimeoutError: If execution exceeds timeout.
    """
    ...
```

### YAML (Agent Contracts)

```yaml
# Use 2-space indentation
# Quote strings containing special characters
# Use comments to explain non-obvious choices

apiVersion: datamesh.ai/v1
kind: Agent

metadata:
  name: sql-agent  # Lowercase, hyphenated
  version: 1.0.0   # Semantic versioning
```

### Documentation

- Use clear, concise language
- Include code examples
- Keep README files focused
- Use Markdown tables for structured data

---

## Community

### Communication Channels

| Channel | Purpose |
|---------|---------|
| **GitHub Issues** | Bug reports, feature requests |
| **GitHub Discussions** | Questions, ideas, show & tell |
| **Discord** | Real-time chat, community support |
| **Twitter** | Announcements, news |

### Getting Help

- **Search first** — Your question may already be answered
- **Be specific** — Include versions, errors, and context
- **Be patient** — Maintainers are often volunteers
- **Pay it forward** — Help others when you can

### Recognition

Contributors are recognized in:
- The [CONTRIBUTORS](CONTRIBUTORS.md) file
- Release notes
- Community spotlights

---

## Questions?

Don't hesitate to ask! Open an issue, start a discussion, or join Discord.

**Thank you for contributing to DataMesh-AI!** 🎉
