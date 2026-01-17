#!/bin/bash
# ==============================================================================
# DATAMESH.AI - TLS Certificate Generation Script
# ==============================================================================
# Generates self-signed certificates for local development mTLS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERTS_DIR="${SCRIPT_DIR}/../certs"
VALIDITY_DAYS=365

# Certificate subject configuration
CA_SUBJECT="/C=FR/ST=Ile-de-France/L=Paris/O=DATAMESH.AI/OU=Platform/CN=DATAMESH-CA"
ORCHESTRATOR_SUBJECT="/C=FR/ST=Ile-de-France/L=Paris/O=DATAMESH.AI/OU=Platform/CN=orchestrator"
SQL_AGENT_SUBJECT="/C=FR/ST=Ile-de-France/L=Paris/O=DATAMESH.AI/OU=Agents/CN=sql-agent"
CATALOG_AGENT_SUBJECT="/C=FR/ST=Ile-de-France/L=Paris/O=DATAMESH.AI/OU=Agents/CN=catalog-agent"
GOVERNANCE_AGENT_SUBJECT="/C=FR/ST=Ile-de-France/L=Paris/O=DATAMESH.AI/OU=Agents/CN=governance-agent"

echo "=============================================="
echo "DATAMESH.AI Certificate Generation"
echo "=============================================="
echo ""

# Create certs directory if it doesn't exist
mkdir -p "${CERTS_DIR}"
cd "${CERTS_DIR}"

echo "[1/5] Generating Certificate Authority (CA)..."
# Generate CA private key
openssl genrsa -out ca.key 4096 2>/dev/null

# Generate CA certificate
openssl req -new -x509 -days ${VALIDITY_DAYS} \
    -key ca.key \
    -out ca.crt \
    -subj "${CA_SUBJECT}" 2>/dev/null

echo "  - ca.key (private key)"
echo "  - ca.crt (certificate)"

echo ""
echo "[2/5] Generating Orchestrator certificate..."
# Generate orchestrator private key
openssl genrsa -out orchestrator.key 2048 2>/dev/null

# Generate orchestrator CSR
openssl req -new \
    -key orchestrator.key \
    -out orchestrator.csr \
    -subj "${ORCHESTRATOR_SUBJECT}" 2>/dev/null

# Sign orchestrator certificate with CA
openssl x509 -req -days ${VALIDITY_DAYS} \
    -in orchestrator.csr \
    -CA ca.crt \
    -CAkey ca.key \
    -CAcreateserial \
    -out orchestrator.crt 2>/dev/null

rm orchestrator.csr
echo "  - orchestrator.key (private key)"
echo "  - orchestrator.crt (certificate)"

echo ""
echo "[3/5] Generating SQL Agent certificate..."
# Generate sql-agent private key
openssl genrsa -out sql-agent.key 2048 2>/dev/null

# Generate sql-agent CSR
openssl req -new \
    -key sql-agent.key \
    -out sql-agent.csr \
    -subj "${SQL_AGENT_SUBJECT}" 2>/dev/null

# Sign sql-agent certificate with CA
openssl x509 -req -days ${VALIDITY_DAYS} \
    -in sql-agent.csr \
    -CA ca.crt \
    -CAkey ca.key \
    -CAcreateserial \
    -out sql-agent.crt 2>/dev/null

rm sql-agent.csr
echo "  - sql-agent.key (private key)"
echo "  - sql-agent.crt (certificate)"

echo ""
echo "[4/5] Generating Catalog Agent certificate..."
# Generate catalog-agent private key
openssl genrsa -out catalog-agent.key 2048 2>/dev/null

# Generate catalog-agent CSR
openssl req -new \
    -key catalog-agent.key \
    -out catalog-agent.csr \
    -subj "${CATALOG_AGENT_SUBJECT}" 2>/dev/null

# Sign catalog-agent certificate with CA
openssl x509 -req -days ${VALIDITY_DAYS} \
    -in catalog-agent.csr \
    -CA ca.crt \
    -CAkey ca.key \
    -CAcreateserial \
    -out catalog-agent.crt 2>/dev/null

rm catalog-agent.csr
echo "  - catalog-agent.key (private key)"
echo "  - catalog-agent.crt (certificate)"

echo ""
echo "[5/5] Generating Governance Agent certificate..."
# Generate governance-agent private key
openssl genrsa -out governance-agent.key 2048 2>/dev/null

# Generate governance-agent CSR
openssl req -new \
    -key governance-agent.key \
    -out governance-agent.csr \
    -subj "${GOVERNANCE_AGENT_SUBJECT}" 2>/dev/null

# Sign governance-agent certificate with CA
openssl x509 -req -days ${VALIDITY_DAYS} \
    -in governance-agent.csr \
    -CA ca.crt \
    -CAkey ca.key \
    -CAcreateserial \
    -out governance-agent.crt 2>/dev/null

rm governance-agent.csr
echo "  - governance-agent.key (private key)"
echo "  - governance-agent.crt (certificate)"

# Set permissions
chmod 600 *.key
chmod 644 *.crt

echo ""
echo "=============================================="
echo "Certificate generation complete!"
echo "=============================================="
echo ""
echo "Generated files in ${CERTS_DIR}:"
ls -la "${CERTS_DIR}"/*.crt "${CERTS_DIR}"/*.key 2>/dev/null || true
echo ""
echo "Certificates valid for ${VALIDITY_DAYS} days"
