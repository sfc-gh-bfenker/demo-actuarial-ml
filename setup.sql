-- =============================================================================
-- setup.sql  —  Actuarial Pricing Demo
-- =============================================================================
-- Prerequisites: ACCOUNTADMIN privilege; SPCS enabled on account.
--
-- To customize object names: change the SET variables below, then update
-- src/config.py to match. See CUSTOMIZING.md for the full checklist.
--
-- Run order: execute this file top-to-bottom in a Snowflake worksheet.
-- =============================================================================

USE ROLE ACCOUNTADMIN;

-- ⚙️  Object names — edit here if you need custom names.
--     If you change these, also update src/config.py and create-table.sql.
--     See CUSTOMIZING.md for the full list of files to update.
SET db_name    = 'ACTUARIAL_DEMO_DB';
SET schema_name = 'ACTUARIAL_PRICING';
SET wh_name    = 'ACTUARIAL_DEMO_WH';
SET pool_name  = 'ACTUARIAL_DEMO_POOL';
SET role_name  = 'ACTUARIAL_DEMO_ROLE';

-- =============================================================================
-- ①  Database & schema
-- =============================================================================
CREATE DATABASE IF NOT EXISTS IDENTIFIER($db_name);
CREATE SCHEMA   IF NOT EXISTS IDENTIFIER($db_name || '.' || $schema_name);

-- =============================================================================
-- ②  Virtual warehouse
-- =============================================================================
CREATE WAREHOUSE IF NOT EXISTS IDENTIFIER($wh_name)
    WAREHOUSE_SIZE      = XSMALL
    AUTO_SUSPEND        = 60
    AUTO_RESUME         = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- =============================================================================
-- ③  Compute pool  (requires SPCS enabled on the account)
--     Increase MAX_NODES for multi-node distributed training.
-- =============================================================================
CREATE COMPUTE POOL IF NOT EXISTS IDENTIFIER($pool_name)
    MIN_NODES       = 1
    MAX_NODES       = 3
    INSTANCE_FAMILY = CPU_X64_S
    AUTO_SUSPEND_SECS = 300;

-- =============================================================================
-- ④  Role & grants
-- =============================================================================
CREATE ROLE IF NOT EXISTS IDENTIFIER($role_name);

GRANT USAGE ON DATABASE  IDENTIFIER($db_name) TO ROLE IDENTIFIER($role_name);
GRANT ALL   ON SCHEMA    IDENTIFIER($db_name || '.' || $schema_name)
                         TO ROLE IDENTIFIER($role_name);
GRANT USAGE ON WAREHOUSE IDENTIFIER($wh_name) TO ROLE IDENTIFIER($role_name);
GRANT USAGE ON COMPUTE POOL IDENTIFIER($pool_name) TO ROLE IDENTIFIER($role_name);
GRANT CREATE SERVICE ON SCHEMA IDENTIFIER($db_name || '.' || $schema_name)
                         TO ROLE IDENTIFIER($role_name);
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE IDENTIFIER($role_name);

-- Grant the role to the current user so it can be activated immediately.
GRANT ROLE IDENTIFIER($role_name) TO USER CURRENT_USER();

-- =============================================================================
-- ⑤  Stages
-- =============================================================================
USE SCHEMA IDENTIFIER($db_name || '.' || $schema_name);

CREATE STAGE IF NOT EXISTS DATA_STAGE;     -- raw XML upload target
CREATE STAGE IF NOT EXISTS OUTPUT_STAGE;  -- model artefacts / exports
CREATE STAGE IF NOT EXISTS PAYLOAD_STAGE; -- ML Job payload uploads

-- =============================================================================
-- ⑥  Next steps
-- =============================================================================
-- After this script completes, run the following in order:
--
--   Step A — Load data (run locally, ~2 min):
--     python src/load_actuarial_data.py
--     Downloads freMTPL2 from OpenML, converts to XML, uploads to DATA_STAGE.
--     Pass --help for authentication options.
--
--   Step B — Create tables (run in a Snowflake worksheet):
--     Run create-table.sql
--     Parses the staged XML into HOME_POLICY_FREQ and HOME_POLICY_SEV.
--
--   Step C — Run the demo notebook:
--     Open notebooks/actuarial_pricing_demo.ipynb
-- =============================================================================
