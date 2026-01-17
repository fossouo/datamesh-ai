# DataMesh.AI AWS Quickstart
# ==========================
# Connect DataMesh.AI agents to real AWS data (Athena/Glue)

.PHONY: help setup test-connection run-catalog run-sql run-governance run-orchestrator run-demo run-a2a test-a2a stop clean generate-data nl-demo dashboard

# Default target
help:
	@echo "=============================================="
	@echo "  DATAMESH.AI - AWS Quickstart"
	@echo "=============================================="
	@echo ""
	@echo "Prerequisites:"
	@echo "  - AWS CLI configured with valid credentials"
	@echo "  - Access to Glue Catalog and Athena"
	@echo "  - Python 3.11+"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup           - Create venv and install dependencies"
	@echo "  make test-connection - Test AWS connectivity"
	@echo "  make run-catalog     - Start Catalog Agent (Glue)"
	@echo "  make run-sql         - Start SQL Agent (Athena)"
	@echo "  make run-governance  - Start Governance Agent"
	@echo "  make run-orchestrator- Start A2A Orchestrator"
	@echo "  make run-demo        - Run interactive demo (2 agents)"
	@echo "  make run-a2a         - Run full A2A system (4 agents)"
	@echo "  make test-a2a        - Test A2A communication flow"
	@echo "  make stop            - Stop all running agents"
	@echo "  make generate-data   - Generate test data (500 sessions)"
	@echo "  make nl-demo         - Run NL-to-SQL interactive demo"
	@echo "  make dashboard       - Generate HTML dashboard"
	@echo "  make clean           - Remove virtual environment"
	@echo ""
	@echo "Environment variables:"
	@echo "  AWS_REGION          - AWS region (default: eu-west-1)"
	@echo "  ATHENA_DATABASE     - Athena database (default: talki_metrics_dev)"
	@echo "  ATHENA_OUTPUT       - S3 path for query results"
	@echo ""

# Setup virtual environment
setup:
	@echo "[1/3] Creating Python virtual environment..."
	python3 -m venv .venv
	@echo "[2/3] Upgrading pip..."
	.venv/bin/pip install --upgrade pip
	@echo "[3/3] Installing dependencies..."
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Setup complete!"
	@echo "Run 'make test-connection' to verify AWS access."

# Test AWS connection
test-connection:
	@echo "Testing AWS connection..."
	.venv/bin/python scripts/test_aws_connection.py

# Run Catalog Agent with Glue integration
run-catalog:
	@echo "Starting Catalog Agent (Glue)..."
	@echo "Database filter: talki_*"
	.venv/bin/python agents/catalog_agent_aws.py

# Run SQL Agent with Athena
run-sql:
	@echo "Starting SQL Agent (Athena)..."
	@echo "Database: $${ATHENA_DATABASE:-talki_metrics_dev}"
	.venv/bin/python agents/sql_agent_aws.py

# Run interactive demo
run-demo: .venv
	@echo "=============================================="
	@echo "  DATAMESH.AI - AWS Demo"
	@echo "=============================================="
	@echo ""
	@echo "Starting agents in background..."
	@.venv/bin/python agents/catalog_agent_aws.py &
	@sleep 2
	@.venv/bin/python agents/sql_agent_aws.py &
	@sleep 2
	@echo ""
	@echo "Agents started:"
	@echo "  - Catalog Agent: http://localhost:8082"
	@echo "  - SQL Agent: http://localhost:8081"
	@echo ""
	@echo "Testing Catalog Agent..."
	@curl -s http://localhost:8082/health | python3 -m json.tool
	@echo ""
	@echo "Testing SQL Agent..."
	@curl -s http://localhost:8081/health | python3 -m json.tool
	@echo ""
	@echo "=============================================="
	@echo "Demo Ready! Try these commands:"
	@echo "=============================================="
	@echo ""
	@echo "# List available datasets:"
	@echo "curl -s http://localhost:8082/datasets | jq"
	@echo ""
	@echo "# Resolve session_logs schema:"
	@echo 'curl -s -X POST http://localhost:8082 -H "Content-Type: application/json" \'
	@echo '  -d '\''{"capability": "catalog.resolve", "payload": {"dataset": "catalog://talki_metrics_dev.session_logs"}}'\'' | jq'
	@echo ""
	@echo "# Generate SQL for a question:"
	@echo 'curl -s -X POST http://localhost:8081 -H "Content-Type: application/json" \'
	@echo '  -d '\''{"capability": "sql.generate", "payload": {"question": "Show costs by language"}}'\'' | jq'
	@echo ""
	@echo "# Execute SQL query:"
	@echo 'curl -s -X POST http://localhost:8081 -H "Content-Type: application/json" \'
	@echo '  -d '\''{"capability": "sql.execute", "payload": {"sql": "SELECT language, COUNT(*) as sessions FROM session_logs GROUP BY language LIMIT 10"}}'\'' | jq'
	@echo ""
	@echo "Press Ctrl+C to stop agents..."
	@trap 'kill $$(jobs -p) 2>/dev/null' EXIT; wait

# Interactive query session
query:
	@echo "Enter SQL query (or 'exit' to quit):"
	@while true; do \
		read -p "sql> " query; \
		if [ "$$query" = "exit" ]; then break; fi; \
		curl -s -X POST http://localhost:8081 \
			-H "Content-Type: application/json" \
			-d "{\"capability\": \"sql.execute\", \"payload\": {\"sql\": \"$$query\"}}" \
			| python3 -m json.tool; \
	done

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf logs/
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	@echo "Clean complete."

# Run Governance Agent
run-governance:
	@echo "Starting Governance Agent..."
	.venv/bin/python agents/governance_agent_aws.py

# Run A2A Orchestrator
run-orchestrator:
	@echo "Starting A2A Orchestrator..."
	.venv/bin/python orchestrator/orchestrator_a2a.py

# Run full A2A system (all 4 components)
run-a2a: .venv
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║            DATAMESH.AI - A2A System Startup                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "[1/4] Starting Catalog Agent (Glue)..."
	@STAGE=prod .venv/bin/python agents/catalog_agent_aws.py > logs/catalog.log 2>&1 &
	@sleep 2
	@echo "[2/4] Starting SQL Agent (Athena)..."
	@ATHENA_DATABASE=talki_metrics_prod .venv/bin/python agents/sql_agent_aws.py > logs/sql.log 2>&1 &
	@sleep 2
	@echo "[3/4] Starting Governance Agent..."
	@.venv/bin/python agents/governance_agent_aws.py > logs/governance.log 2>&1 &
	@sleep 2
	@echo "[4/4] Starting A2A Orchestrator..."
	@.venv/bin/python orchestrator/orchestrator_a2a.py > logs/orchestrator.log 2>&1 &
	@sleep 3
	@echo ""
	@echo "Checking agent health..."
	@curl -s http://localhost:8080/health | python3 -m json.tool || echo "Orchestrator not ready"
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║  All agents started! Use 'make test-a2a' to test A2A flow        ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  Endpoints:                                                      ║"
	@echo "║    Orchestrator:    http://localhost:8080                        ║"
	@echo "║    SQL Agent:       http://localhost:8081                        ║"
	@echo "║    Catalog Agent:   http://localhost:8082                        ║"
	@echo "║    Governance Agent: http://localhost:8083                       ║"
	@echo "║                                                                  ║"
	@echo "║  Logs: ./logs/                                                   ║"
	@echo "║  Stop: make stop                                                 ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

# Test A2A communication flow
test-a2a: .venv
	@echo "Testing A2A communication..."
	@mkdir -p logs
	.venv/bin/python scripts/test_a2a_flow.py

# Stop all agents
stop:
	@echo "Stopping all agents..."
	@pkill -f "catalog_agent_aws.py" 2>/dev/null || true
	@pkill -f "sql_agent_aws.py" 2>/dev/null || true
	@pkill -f "governance_agent_aws.py" 2>/dev/null || true
	@pkill -f "orchestrator_a2a.py" 2>/dev/null || true
	@echo "All agents stopped."

# Generate test data
generate-data: .venv
	@echo "Generating test data (500 sessions)..."
	ATHENA_DATABASE=talki_metrics_prod .venv/bin/python scripts/generate_test_data.py --sessions 500 --days 30

# NL-to-SQL interactive demo
nl-demo: .venv
	@echo "Starting NL-to-SQL interactive demo..."
	ATHENA_DATABASE=talki_metrics_prod .venv/bin/python scripts/nl_query_demo.py

# Generate HTML dashboard
dashboard: .venv
	@echo "Generating dashboard..."
	@mkdir -p dashboard
	ATHENA_DATABASE=talki_metrics_prod .venv/bin/python dashboard/generate_report.py
	@echo ""
	@echo "Dashboard ready! Open: file://$(PWD)/dashboard/index.html"

# Check if venv exists
.venv:
	@echo "Virtual environment not found. Run 'make setup' first."
	@exit 1
