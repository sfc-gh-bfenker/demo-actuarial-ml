-- =============================================================================
-- create-table.sql
-- Load raw XML data from stage and create HOME_POLICY_FREQ / HOME_POLICY_SEV.
--
-- Prerequisites:
--   1. XML files have been uploaded to OUTPUT_STAGE/inbound/ via
--      load_actuarial_data.py (or manually via PUT).
--   2. Update the USE statements below to match your environment if needed.
--
-- Run order: execute this file top-to-bottom in a Snowflake worksheet or
-- via SnowSQL / the Snowflake CLI.
-- =============================================================================

USE DATABASE COUNTRY_ML;
USE SCHEMA   ACTUARIAL_PRICING;
USE ROLE     ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;

-- =============================================================================
-- 1. File format
--    STRIP_OUTER_ELEMENT = TRUE removes <PolicyFeed> / <ClaimFeed> so each
--    direct child element (<Policy>, <Claim>) becomes a separate VARIANT row.
-- =============================================================================

CREATE FILE FORMAT IF NOT EXISTS XML_FF
    TYPE               = XML
    STRIP_OUTER_ELEMENT = TRUE;

-- =============================================================================
-- 2. Raw staging tables (one VARIANT row per XML element)
-- =============================================================================

CREATE OR REPLACE TABLE RAW_POLICY_XML (SRC VARIANT);
CREATE OR REPLACE TABLE RAW_CLAIM_XML  (SRC VARIANT);

-- =============================================================================
-- 3. Load from stage
--    PATTERN filters each COPY INTO to only its matching file so both tables
--    can be loaded from the same stage directory without cross-contamination.
-- =============================================================================

COPY INTO RAW_POLICY_XML
FROM @DATA_STAGE/
PATTERN      = '.*policy_freq.*'
FILE_FORMAT  = (FORMAT_NAME = XML_FF)
PURGE        = FALSE;

COPY INTO RAW_CLAIM_XML
FROM @DATA_STAGE/
PATTERN      = '.*policy_sev.*'
FILE_FORMAT  = (FORMAT_NAME = XML_FF)
PURGE        = FALSE;

-- =============================================================================
-- 4. HOME_POLICY_FREQ  (~678K rows)
--    XMLGET(node, 'Tag'):"$" extracts the text content of the element.
--    The "$ path accessor must be double-quoted in Snowflake SQL.
--    Nested elements use chained calls: XMLGET(XMLGET(src, 'Risk'), 'X').
-- =============================================================================

CREATE OR REPLACE TABLE HOME_POLICY_FREQ AS
SELECT
    XMLGET(SRC, 'PolicyId'):"$"::BIGINT          AS POLICY_ID,
    XMLGET(SRC, 'Exposure'):"$"::FLOAT           AS EXPOSURE,
    XMLGET(SRC, 'PolicyholderAge'):"$"::INTEGER  AS POLICYHOLDER_AGE,
    XMLGET(SRC, 'LossHistoryScore'):"$"::FLOAT   AS LOSS_HISTORY_SCORE,
    XMLGET(SRC, 'PopulationDensity'):"$"::FLOAT  AS POPULATION_DENSITY,
    XMLGET(SRC, 'RegionCode'):"$"::VARCHAR        AS REGION_CODE,
    XMLGET(XMLGET(SRC, 'Risk'), 'TerritoryCode'):"$"::VARCHAR       AS TERRITORY_CODE,
    XMLGET(XMLGET(SRC, 'Risk'), 'ConstructionType'):"$"::VARCHAR    AS CONSTRUCTION_TYPE,
    XMLGET(XMLGET(SRC, 'Risk'), 'ConstructionQuality'):"$"::INTEGER AS CONSTRUCTION_QUALITY,
    XMLGET(XMLGET(SRC, 'Risk'), 'PropertyAge'):"$"::INTEGER         AS PROPERTY_AGE,
    XMLGET(XMLGET(SRC, 'Risk'), 'OccupancyType'):"$"::VARCHAR       AS OCCUPANCY_TYPE,
    XMLGET(XMLGET(SRC, 'Claims'), 'ClaimCount'):"$"::INTEGER        AS CLAIM_COUNT
FROM RAW_POLICY_XML;

SELECT COUNT(*) AS home_policy_freq_rows FROM HOME_POLICY_FREQ;  -- expect ~678,013

-- =============================================================================
-- 5. HOME_POLICY_SEV  (~26K rows)
-- =============================================================================

CREATE OR REPLACE TABLE HOME_POLICY_SEV AS
SELECT
    XMLGET(SRC, 'PolicyId'):"$"::BIGINT    AS POLICY_ID,
    XMLGET(SRC, 'ClaimAmount'):"$"::FLOAT  AS CLAIM_AMOUNT
FROM RAW_CLAIM_XML;

SELECT COUNT(*) AS home_policy_sev_rows FROM HOME_POLICY_SEV;    -- expect ~26,639
