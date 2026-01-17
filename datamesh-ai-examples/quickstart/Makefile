# DATAMESH.AI Quickstart Makefile
# ================================
# This Makefile provides targets for setting up and running the local A2A demo

.PHONY: bootstrap run-demo down clean help

# Default target
.DEFAULT_GOAL := help

# Configuration
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DOCKER_COMPOSE := docker compose
LOGS_DIR := logs
BOOTLOGS_DIR := bootlogs
CERTS_DIR := certs
AUDIT_DIR := $(LOGS_DIR)/audit

# Colors for terminal output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "DATAMESH.AI Quickstart"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:"
	@echo "  make bootstrap  # Set up environment"
	@echo "  make run-demo   # Run the A2A demo"
	@echo "  make down       # Stop all services"

# =============================================================================
# Bootstrap Target
# =============================================================================

bootstrap: ## Create venv, install deps, generate certs, start docker
	@echo "$(GREEN)[1/5] Creating Python virtual environment...$(NC)"
	@mkdir -p $(BOOTLOGS_DIR) $(LOGS_DIR) $(AUDIT_DIR) $(CERTS_DIR)
	@python3 -m venv $(VENV) 2>&1 | tee $(BOOTLOGS_DIR)/venv.log
	@echo "$(GREEN)[2/5] Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip 2>&1 | tee $(BOOTLOGS_DIR)/pip-upgrade.log
	@$(PIP) install -r requirements.txt 2>&1 | tee $(BOOTLOGS_DIR)/pip-install.log
	@echo "$(GREEN)[3/5] Generating TLS certificates...$(NC)"
	@chmod +x ./scripts/generate-certs.sh
	@./scripts/generate-certs.sh 2>&1 | tee $(BOOTLOGS_DIR)/certs.log
	@echo "$(GREEN)[4/5] Loading baseline policies...$(NC)"
	@echo "  - policies/rgpd.yaml"
	@echo "  - policies/data-retention.yaml"
	@echo "$(GREEN)[5/5] Starting Docker Compose stack...$(NC)"
	@$(DOCKER_COMPOSE) up -d 2>&1 | tee $(BOOTLOGS_DIR)/docker.log
	@echo ""
	@echo "$(GREEN)Bootstrap complete!$(NC)"
	@echo "Run 'make run-demo' to start the A2A demonstration."

# =============================================================================
# Run Demo Target
# =============================================================================

run-demo: ## Start orchestrator + agents, send sample request
	@echo "$(GREEN)[1/4] Starting orchestrator...$(NC)"
	@$(PYTHON) orchestrator/main.py --config orchestrator/config.yaml &
	@sleep 2
	@echo "$(GREEN)[2/4] Starting agents...$(NC)"
	@$(PYTHON) agents/sql-agent/main.py --config agents/sql-agent/agent.yaml &
	@$(PYTHON) agents/catalog-agent/main.py --config agents/catalog-agent/agent.yaml &
	@$(PYTHON) agents/governance-agent/main.py --config agents/governance-agent/agent.yaml &
	@sleep 3
	@echo "$(GREEN)[3/4] Sending sample request...$(NC)"
	@echo ""
	@cat requests/monthly_churn.sql.json
	@echo ""
	@echo ""
	@$(PYTHON) scripts/send-request.py requests/monthly_churn.sql.json 2>&1 | tee $(LOGS_DIR)/demo-run.log
	@echo ""
	@echo "$(GREEN)[4/4] Demo complete!$(NC)"
	@echo ""
	@echo "Check the following for details:"
	@echo "  - Orchestrator logs: $(LOGS_DIR)/orchestrator.log"
	@echo "  - Audit trails: $(AUDIT_DIR)/"
	@echo "  - OpenTelemetry: http://localhost:4318 (if enabled)"

run-demo-async: ## Run demo in async mode via Kafka
	@echo "$(YELLOW)Running in ASYNC mode (Kafka-backed)...$(NC)"
	@ASYNC=1 $(MAKE) run-demo

# =============================================================================
# Down Target
# =============================================================================

down: ## Stop all services
	@echo "$(YELLOW)Stopping agents...$(NC)"
	@pkill -f "agents/sql-agent/main.py" 2>/dev/null || true
	@pkill -f "agents/catalog-agent/main.py" 2>/dev/null || true
	@pkill -f "agents/governance-agent/main.py" 2>/dev/null || true
	@echo "$(YELLOW)Stopping orchestrator...$(NC)"
	@pkill -f "orchestrator/main.py" 2>/dev/null || true
	@echo "$(YELLOW)Stopping Docker Compose stack...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)All services stopped.$(NC)"
ifdef CLEAN_AUDIT
	@echo "$(YELLOW)Cleaning audit logs (CLEAN_AUDIT=1)...$(NC)"
	@rm -rf $(AUDIT_DIR)/*
endif

# =============================================================================
# Clean Target
# =============================================================================

clean: down ## Remove transient files (venv, logs, certs, docker volumes)
	@echo "$(RED)Cleaning transient files...$(NC)"
	@rm -rf $(VENV)
	@rm -rf $(LOGS_DIR)/*.log
	@rm -rf $(BOOTLOGS_DIR)/*.log
	@rm -rf $(CERTS_DIR)/*.pem $(CERTS_DIR)/*.key $(CERTS_DIR)/*.crt
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@$(DOCKER_COMPOSE) down -v 2>/dev/null || true
	@echo "$(GREEN)Clean complete.$(NC)"

clean-all: clean ## Remove everything including audit logs
	@echo "$(RED)Removing audit logs...$(NC)"
	@rm -rf $(AUDIT_DIR)/*
	@echo "$(GREEN)Full clean complete.$(NC)"

# =============================================================================
# Utility Targets
# =============================================================================

status: ## Show status of all services
	@echo "$(GREEN)Docker Compose Status:$(NC)"
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "$(GREEN)Running Processes:$(NC)"
	@ps aux | grep -E "(orchestrator|sql-agent|catalog-agent|governance-agent)" | grep -v grep || echo "  No agents running"

logs: ## Tail all logs
	@tail -f $(LOGS_DIR)/*.log

logs-orchestrator: ## Tail orchestrator logs
	@tail -f $(LOGS_DIR)/orchestrator.log

logs-audit: ## List audit log files
	@ls -la $(AUDIT_DIR)/

verify: ## Verify prerequisites are installed
	@echo "Checking prerequisites..."
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)Docker is not installed$(NC)"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "$(RED)Python 3 is not installed$(NC)"; exit 1; }
	@echo "$(GREEN)All prerequisites met!$(NC)"
