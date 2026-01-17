-- ==============================================================================
-- DATAMESH.AI - Catalog Database Initialization
-- ==============================================================================
-- Creates the mock catalog schema for the quickstart demo

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- Tables Catalog
-- ==============================================================================

CREATE TABLE IF NOT EXISTS catalog_tables (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    schema_name VARCHAR(255) DEFAULT 'public',
    description TEXT,
    owner VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- ==============================================================================
-- Columns Catalog
-- ==============================================================================

CREATE TABLE IF NOT EXISTS catalog_columns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_id UUID REFERENCES catalog_tables(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    data_type VARCHAR(100) NOT NULL,
    nullable BOOLEAN DEFAULT true,
    description TEXT,
    classification VARCHAR(50) DEFAULT 'UNCLASSIFIED',
    pii_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(table_id, name)
);

-- ==============================================================================
-- Relationships
-- ==============================================================================

CREATE TABLE IF NOT EXISTS catalog_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_table_id UUID REFERENCES catalog_tables(id) ON DELETE CASCADE,
    from_column VARCHAR(255) NOT NULL,
    to_table_id UUID REFERENCES catalog_tables(id) ON DELETE CASCADE,
    to_column VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(50) DEFAULT 'many-to-one',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================================================
-- Lineage
-- ==============================================================================

CREATE TABLE IF NOT EXISTS catalog_lineage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_id UUID REFERENCES catalog_tables(id) ON DELETE CASCADE,
    upstream_source VARCHAR(255) NOT NULL,
    downstream_target VARCHAR(255),
    transformation_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================================================
-- Insert Demo Data
-- ==============================================================================

-- Insert tables
INSERT INTO catalog_tables (name, description, owner, tags) VALUES
    ('revenue', 'Monthly revenue records by region', 'finance-team', ARRAY['pii-free', 'revenue', 'metrics']),
    ('regions', 'Geographic regions for business operations', 'operations-team', ARRAY['reference', 'geography']),
    ('time_periods', 'Time dimension table for analytics', 'data-platform', ARRAY['reference', 'time']),
    ('customers', 'Customer master data', 'crm-team', ARRAY['pii', 'customer', 'gdpr-relevant']),
    ('transactions', 'Individual transaction records', 'finance-team', ARRAY['transactional', 'high-volume'])
ON CONFLICT (name) DO NOTHING;

-- Insert columns for revenue table
INSERT INTO catalog_columns (table_id, name, data_type, description, classification)
SELECT id, 'id', 'UUID', 'Primary key', 'INTERNAL'
FROM catalog_tables WHERE name = 'revenue'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification)
SELECT id, 'amount', 'DECIMAL(15,2)', 'Revenue amount in EUR', 'CONFIDENTIAL'
FROM catalog_tables WHERE name = 'revenue'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification)
SELECT id, 'region_id', 'UUID', 'Foreign key to regions', 'INTERNAL'
FROM catalog_tables WHERE name = 'revenue'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification)
SELECT id, 'period_id', 'UUID', 'Foreign key to time_periods', 'INTERNAL'
FROM catalog_tables WHERE name = 'revenue'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification)
SELECT id, 'created_at', 'TIMESTAMP', 'Record creation timestamp', 'INTERNAL'
FROM catalog_tables WHERE name = 'revenue'
ON CONFLICT DO NOTHING;

-- Insert columns for regions table
INSERT INTO catalog_columns (table_id, name, data_type, description, classification)
SELECT id, 'id', 'UUID', 'Primary key', 'INTERNAL'
FROM catalog_tables WHERE name = 'regions'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification)
SELECT id, 'name', 'VARCHAR(255)', 'Region name', 'PUBLIC'
FROM catalog_tables WHERE name = 'regions'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification)
SELECT id, 'country', 'VARCHAR(2)', 'ISO country code', 'PUBLIC'
FROM catalog_tables WHERE name = 'regions'
ON CONFLICT DO NOTHING;

-- Insert columns for customers table (with PII)
INSERT INTO catalog_columns (table_id, name, data_type, description, classification, pii_type)
SELECT id, 'id', 'UUID', 'Primary key', 'INTERNAL', NULL
FROM catalog_tables WHERE name = 'customers'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification, pii_type)
SELECT id, 'name', 'VARCHAR(255)', 'Customer full name', 'PII', 'NAME'
FROM catalog_tables WHERE name = 'customers'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification, pii_type)
SELECT id, 'email', 'VARCHAR(255)', 'Customer email address', 'PII', 'EMAIL'
FROM catalog_tables WHERE name = 'customers'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_columns (table_id, name, data_type, description, classification, pii_type)
SELECT id, 'phone', 'VARCHAR(50)', 'Customer phone number', 'PII', 'PHONE'
FROM catalog_tables WHERE name = 'customers'
ON CONFLICT DO NOTHING;

-- Insert relationships
INSERT INTO catalog_relationships (from_table_id, from_column, to_table_id, to_column, relationship_type)
SELECT r.id, 'region_id', reg.id, 'id', 'many-to-one'
FROM catalog_tables r, catalog_tables reg
WHERE r.name = 'revenue' AND reg.name = 'regions'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_relationships (from_table_id, from_column, to_table_id, to_column, relationship_type)
SELECT r.id, 'period_id', tp.id, 'id', 'many-to-one'
FROM catalog_tables r, catalog_tables tp
WHERE r.name = 'revenue' AND tp.name = 'time_periods'
ON CONFLICT DO NOTHING;

-- Insert lineage
INSERT INTO catalog_lineage (table_id, upstream_source, transformation_type)
SELECT id, 'raw_transactions', 'aggregation'
FROM catalog_tables WHERE name = 'revenue'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_lineage (table_id, upstream_source, transformation_type)
SELECT id, 'erp_system', 'etl'
FROM catalog_tables WHERE name = 'revenue'
ON CONFLICT DO NOTHING;

INSERT INTO catalog_lineage (table_id, downstream_target, transformation_type)
SELECT id, 'revenue_dashboard', 'view'
FROM catalog_tables WHERE name = 'revenue'
ON CONFLICT DO NOTHING;

-- ==============================================================================
-- Create Views for Common Queries
-- ==============================================================================

CREATE OR REPLACE VIEW v_table_columns AS
SELECT
    t.name as table_name,
    t.description as table_description,
    t.owner,
    t.tags,
    c.name as column_name,
    c.data_type,
    c.classification,
    c.pii_type
FROM catalog_tables t
LEFT JOIN catalog_columns c ON t.id = c.table_id
ORDER BY t.name, c.name;

CREATE OR REPLACE VIEW v_table_relationships AS
SELECT
    ft.name as from_table,
    r.from_column,
    tt.name as to_table,
    r.to_column,
    r.relationship_type
FROM catalog_relationships r
JOIN catalog_tables ft ON r.from_table_id = ft.id
JOIN catalog_tables tt ON r.to_table_id = tt.id;

-- ==============================================================================
-- Grant Permissions
-- ==============================================================================

GRANT SELECT ON ALL TABLES IN SCHEMA public TO datamesh;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO datamesh;

-- Done!
