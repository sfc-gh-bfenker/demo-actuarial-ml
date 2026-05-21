"""
Load freMTPL2 (French Motor TPL) dataset into Snowflake as a homeowners insurance dataset.

The `freMTPL2 <https://www.openml.org/d/41214>`_ dataset is a French motor
third-party liability portfolio (~678 K policies) published on OpenML.  Column
names and value domains are remapped here to match a homeowners insurance
context so the demo is directly relatable to property/casualty actuaries.

The entire dataset is serialised to XML and loaded into Snowflake via
``COPY INTO``, mirroring how a carrier receives Ratabase / PolicyPro rating
extracts in practice.  The final ``HOME_POLICY_FREQ`` and ``HOME_POLICY_SEV``
tables are created by pure-SQL ``XMLGET`` parsing — no ``write_pandas`` step.

Usage
-----
::

    python load_actuarial_data.py [OPTIONS]

    All connection/target parameters default to the demo values but can be
    overridden via CLI flags or environment variables:

      --connection        SNOWFLAKE_CONNECTION   named connection from connections.toml
      --account           SNOWFLAKE_ACCOUNT
      --user              SNOWFLAKE_USER
      --role              SNOWFLAKE_ROLE
      --warehouse         SNOWFLAKE_WAREHOUSE
      --database          SNOWFLAKE_DATABASE
      --schema            SNOWFLAKE_SCHEMA
      --private-key-file  SNOWFLAKE_PRIVATE_KEY_FILE

Tables written
--------------
``<database>.<schema>.RAW_POLICY_XML``
    ~678 K rows — raw VARIANT column; one row per ``<Policy>`` element.

``<database>.<schema>.RAW_CLAIM_XML``
    ~26 K rows — raw VARIANT column; one row per ``<Claim>`` element.

``<database>.<schema>.HOME_POLICY_FREQ``
    ~678 K rows — policy-level frequency data parsed from ``RAW_POLICY_XML``
    using ``XMLGET``.

``<database>.<schema>.HOME_POLICY_SEV``
    ~26 K rows — claim-level severity data parsed from ``RAW_CLAIM_XML``
    using ``XMLGET``.

Snowflake features used
-----------------------
Python Connector:
    https://docs.snowflake.com/en/developer-guide/python-connector/python-connector
Named connections (connections.toml):
    https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#using-a-connection-string
Key-pair authentication:
    https://docs.snowflake.com/en/user-guide/key-pair-auth
PUT command (local file → stage):
    https://docs.snowflake.com/en/sql-reference/sql/put
COPY INTO <table> (stage → table):
    https://docs.snowflake.com/en/sql-reference/sql/copy-into-table
XML file format / STRIP_OUTER_ELEMENT:
    https://docs.snowflake.com/en/sql-reference/sql/create-file-format
XMLGET function (semi-structured XML parsing):
    https://docs.snowflake.com/en/sql-reference/functions/xmlget
VARIANT semi-structured data type:
    https://docs.snowflake.com/en/sql-reference/data-types-semistructured
"""

import argparse
import os
import ssl
import tempfile

import pandas as pd
import snowflake.connector
from config import (
    DATABASE as DEFAULT_DATABASE,
    SCHEMA as DEFAULT_SCHEMA,
    ROLE as DEFAULT_ROLE,
    WAREHOUSE as DEFAULT_WAREHOUSE,
    DEFAULT_ACCOUNT,
    DEFAULT_USER,
    DEFAULT_PRIVATE_KEY_FILE,
)

# ── SSL patch (corporate certificate environment) ─────────────────────────────
# Some corporate networks intercept HTTPS with a custom CA.  Disabling
# verification allows ``fetch_openml`` to reach openml.org.  Remove this line
# if your network does not require it.
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_openml  # noqa: E402 (import after ssl patch)

# ── Connection defaults ───────────────────────────────────────────────────────
# Imported from config.py — edit that file to change environment defaults.
# Override any value via the corresponding CLI flag or environment variable.

# ── freMTPL2 → homeowners column rename maps ──────────────────────────────────
# The source dataset uses French motor insurance terminology.  Columns are
# renamed to homeowners equivalents so that the demo is relatable to property
# actuaries without changing any data values.
FREQ_RENAME = {
    "IDpol": "POLICY_ID",
    "ClaimNb": "CLAIM_COUNT",
    "Exposure": "EXPOSURE",
    "Area": "TERRITORY_CODE",
    "VehPower": "CONSTRUCTION_QUALITY",
    "VehAge": "PROPERTY_AGE",
    "DrivAge": "POLICYHOLDER_AGE",
    "BonusMalus": "LOSS_HISTORY_SCORE",
    "VehBrand": "CONSTRUCTION_TYPE",
    "VehGas": "OCCUPANCY_TYPE",
    "Density": "POPULATION_DENSITY",
    "Region": "REGION_CODE",
}

SEV_RENAME = {
    "IDpol": "POLICY_ID",
    "ClaimAmount": "CLAIM_AMOUNT",
}

# Maps freMTPL2 vehicle brand codes to ISO construction class names.
# ISO construction classes are the standard rating factor for homeowners
# policies in U.S. personal lines pricing.
CONSTRUCTION_TYPE_MAP = {
    "B1": "Frame",
    "B2": "Masonry",
    "B3": "Superior Masonry",
    "B4": "Non-Combustible",
    "B5": "Masonry Non-Combustible",
    "B6": "Modified Fire Resistive",
    "B10": "Fire Resistive",
    "B11": "Log",
    "B12": "Superior Frame",
    "B13": "Manufactured Home",
    "B14": "Mixed Construction",
}

# Maps freMTPL2 fuel type to homeowners occupancy type.
OCCUPANCY_TYPE_MAP = {
    "Regular": "Owner-Occupied",
    "Diesel": "Tenant-Occupied",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments, falling back to environment variables then defaults."""
    parser = argparse.ArgumentParser(
        description="Load freMTPL2 dataset into Snowflake as homeowners insurance data."
    )
    parser.add_argument(
        "--connection",
        default=os.environ.get("SNOWFLAKE_CONNECTION"),
        help="Named connection from connections.toml (env: SNOWFLAKE_CONNECTION).",
    )
    parser.add_argument(
        "--account",
        default=os.environ.get("SNOWFLAKE_ACCOUNT", DEFAULT_ACCOUNT),
        help="Snowflake account identifier (env: SNOWFLAKE_ACCOUNT).",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("SNOWFLAKE_USER", DEFAULT_USER),
        help="Snowflake username (env: SNOWFLAKE_USER).",
    )
    parser.add_argument(
        "--role",
        default=os.environ.get("SNOWFLAKE_ROLE", DEFAULT_ROLE),
        help="Snowflake role (env: SNOWFLAKE_ROLE).",
    )
    parser.add_argument(
        "--warehouse",
        default=os.environ.get("SNOWFLAKE_WAREHOUSE", DEFAULT_WAREHOUSE),
        help="Snowflake warehouse (env: SNOWFLAKE_WAREHOUSE).",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("SNOWFLAKE_DATABASE", DEFAULT_DATABASE),
        help="Target database (env: SNOWFLAKE_DATABASE).",
    )
    parser.add_argument(
        "--schema",
        default=os.environ.get("SNOWFLAKE_SCHEMA", DEFAULT_SCHEMA),
        help="Target schema (env: SNOWFLAKE_SCHEMA).",
    )
    parser.add_argument(
        "--private-key-file",
        default=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE", DEFAULT_PRIVATE_KEY_FILE),
        dest="private_key_file",
        help="Path to RSA private key file for JWT auth (env: SNOWFLAKE_PRIVATE_KEY_FILE).",
    )
    return parser.parse_args()


def download_datasets():
    """Download the freMTPL2 frequency and severity tables from OpenML.

    Source: French Motor Third-Party Liability dataset, originally published by
    Charpentier (2014) and hosted on OpenML:

    - Frequency table (data_id=41214): ~678 K policies, one row per policy.
      https://www.openml.org/d/41214
    - Severity table (data_id=41215): ~26 K claims, one row per claim.
      https://www.openml.org/d/41215

    ``fetch_openml`` caches the downloaded ARFF files in
    ``~/scikit_learn_data/`` after the first call, so subsequent runs are fast.

    Returns:
        Tuple ``(freq_df, sev_df)`` of raw pandas DataFrames before any
        column renaming or type coercion.
    """
    print("Downloading freMTPL2freq (data_id=41214)...")
    freq = fetch_openml(data_id=41214, as_frame=True, parser="auto")
    freq_df = freq.frame.copy()
    print(f"  freq shape: {freq_df.shape}")

    print("Downloading freMTPL2sev (data_id=41215)...")
    sev = fetch_openml(data_id=41215, as_frame=True, parser="auto")
    sev_df = sev.frame.copy()
    print(f"  sev shape:  {sev_df.shape}")

    return freq_df, sev_df


def transform_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and retype the frequency DataFrame for the homeowners demo context.

    Column mapping (freMTPL2 source → homeowners target):

    +--------------------+---------------------------+------------------------------------------+
    | Source column      | Target column             | Notes                                    |
    +====================+===========================+==========================================+
    | IDpol              | POLICY_ID                 | Unique policy identifier                 |
    | ClaimNb            | CLAIM_COUNT               | Number of claims in the exposure period  |
    | Exposure           | EXPOSURE                  | Fraction of a policy-year (0–1]          |
    | Area               | TERRITORY_CODE            | Geographic territory rating factor       |
    | VehPower           | CONSTRUCTION_QUALITY      | Ordinal quality/power rating             |
    | VehAge             | PROPERTY_AGE              | Age of the insured property in years     |
    | DrivAge            | POLICYHOLDER_AGE          | Age of the policyholder in years         |
    | BonusMalus         | LOSS_HISTORY_SCORE        | Experience modifier (≈100 = neutral)     |
    | VehBrand           | CONSTRUCTION_TYPE         | ISO construction class (see lookup map)  |
    | VehGas             | OCCUPANCY_TYPE            | Owner-Occupied / Tenant-Occupied         |
    | Density            | POPULATION_DENSITY        | Population per km² of the municipality  |
    | Region             | REGION_CODE               | French administrative region code        |
    +--------------------+---------------------------+------------------------------------------+

    The ARFF format used by OpenML injects single-quote wrappers around some
    string values (e.g. ``"'Regular'"`` instead of ``"Regular"``); these are
    stripped before the value maps are applied.

    Args:
        df: Raw frequency DataFrame from ``download_datasets()``.

    Returns:
        Cleaned and renamed DataFrame ready for XML serialisation.
    """
    df = df.rename(columns=FREQ_RENAME)

    # Strip ARFF-injected single quotes, then map to homeowners domain values.
    # VehGas arrives as object dtype with values like "'Regular'".
    # VehBrand arrives as categorical dtype and is already clean.
    df["OCCUPANCY_TYPE"] = (
        df["OCCUPANCY_TYPE"].astype(str).str.strip("'").map(OCCUPANCY_TYPE_MAP)
    )
    df["CONSTRUCTION_TYPE"] = (
        df["CONSTRUCTION_TYPE"].astype(str).str.strip("'").map(CONSTRUCTION_TYPE_MAP)
    )

    # Explicit dtype coercion ensures XML serialisation produces correct
    # Python types (int vs float).
    df["POLICY_ID"] = df["POLICY_ID"].astype("int64")
    df["CLAIM_COUNT"] = df["CLAIM_COUNT"].astype("int64")
    df["EXPOSURE"] = df["EXPOSURE"].astype("float64")
    df["CONSTRUCTION_QUALITY"] = df["CONSTRUCTION_QUALITY"].astype("int64")
    df["PROPERTY_AGE"] = df["PROPERTY_AGE"].astype("int64")
    df["POLICYHOLDER_AGE"] = df["POLICYHOLDER_AGE"].astype("int64")
    df["LOSS_HISTORY_SCORE"] = df["LOSS_HISTORY_SCORE"].astype("float64")
    df["POPULATION_DENSITY"] = df["POPULATION_DENSITY"].astype("float64")

    return df


def transform_sev(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and retype the severity DataFrame.

    The severity table is claim-level: a single policy may appear multiple
    times if it had more than one claim in the exposure period.  Aggregation
    to policy level (``sum(CLAIM_AMOUNT) GROUP BY POLICY_ID``) happens
    downstream in the feature engineering notebook cell when joining to the
    frequency table.

    Args:
        df: Raw severity DataFrame from ``download_datasets()``.

    Returns:
        Cleaned and renamed DataFrame with columns ``POLICY_ID`` and
        ``CLAIM_AMOUNT``.
    """
    df = df.rename(columns=SEV_RENAME)
    df["POLICY_ID"] = df["POLICY_ID"].astype("int64")
    df["CLAIM_AMOUNT"] = df["CLAIM_AMOUNT"].astype("float64")
    return df


def generate_freq_xml(freq_df: pd.DataFrame) -> str:
    """Serialise the full frequency DataFrame to a ``<PolicyFeed>`` XML string.

    The schema mimics a Ratabase / PolicyPro-style property rating extract,
    with nested ``<Risk>`` and ``<Claims>`` sub-elements to demonstrate
    Snowflake's chained ``XMLGET`` parsing capability.

    Output structure::

        <PolicyFeed>
          <Policy>
            <PolicyId>...</PolicyId>
            <Exposure>...</Exposure>
            <PolicyholderAge>...</PolicyholderAge>
            <LossHistoryScore>...</LossHistoryScore>
            <PopulationDensity>...</PopulationDensity>
            <RegionCode>...</RegionCode>
            <Risk>
              <TerritoryCode>...</TerritoryCode>
              <ConstructionType>...</ConstructionType>
              <ConstructionQuality>...</ConstructionQuality>
              <PropertyAge>...</PropertyAge>
              <OccupancyType>...</OccupancyType>
            </Risk>
            <Claims>
              <ClaimCount>...</ClaimCount>
            </Claims>
          </Policy>
          ...
        </PolicyFeed>

    Uses ``itertuples()`` for ~15× better throughput than ``iterrows()``
    at 678 K rows.  The ``<PolicyFeed>`` root is stripped by
    ``STRIP_OUTER_ELEMENT = TRUE`` in the Snowflake file format so each
    ``<Policy>`` child becomes a separate VARIANT row in ``RAW_POLICY_XML``.

    Args:
        freq_df: Transformed frequency DataFrame (output of ``transform_freq``).

    Returns:
        UTF-8 XML string without an XML declaration header.
    """
    parts = ["<PolicyFeed>\n"]
    for row in freq_df.itertuples(index=False):
        parts.append(
            f"  <Policy>\n"
            f"    <PolicyId>{row.POLICY_ID}</PolicyId>\n"
            f"    <Exposure>{round(row.EXPOSURE, 6)}</Exposure>\n"
            f"    <PolicyholderAge>{row.POLICYHOLDER_AGE}</PolicyholderAge>\n"
            f"    <LossHistoryScore>{round(row.LOSS_HISTORY_SCORE, 4)}</LossHistoryScore>\n"
            f"    <PopulationDensity>{round(row.POPULATION_DENSITY, 4)}</PopulationDensity>\n"
            f"    <RegionCode>{row.REGION_CODE}</RegionCode>\n"
            f"    <Risk>\n"
            f"      <TerritoryCode>{row.TERRITORY_CODE}</TerritoryCode>\n"
            f"      <ConstructionType>{row.CONSTRUCTION_TYPE}</ConstructionType>\n"
            f"      <ConstructionQuality>{row.CONSTRUCTION_QUALITY}</ConstructionQuality>\n"
            f"      <PropertyAge>{row.PROPERTY_AGE}</PropertyAge>\n"
            f"      <OccupancyType>{row.OCCUPANCY_TYPE}</OccupancyType>\n"
            f"    </Risk>\n"
            f"    <Claims>\n"
            f"      <ClaimCount>{row.CLAIM_COUNT}</ClaimCount>\n"
            f"    </Claims>\n"
            f"  </Policy>\n"
        )
    parts.append("</PolicyFeed>")
    return "".join(parts)


def generate_sev_xml(sev_df: pd.DataFrame) -> str:
    """Serialise the full severity DataFrame to a ``<ClaimFeed>`` XML string.

    Output structure::

        <ClaimFeed>
          <Claim>
            <PolicyId>...</PolicyId>
            <ClaimAmount>...</ClaimAmount>
          </Claim>
          ...
        </ClaimFeed>

    The ``<ClaimFeed>`` root is stripped by ``STRIP_OUTER_ELEMENT = TRUE``
    so each ``<Claim>`` child becomes a separate VARIANT row in
    ``RAW_CLAIM_XML``.

    Args:
        sev_df: Transformed severity DataFrame (output of ``transform_sev``).

    Returns:
        UTF-8 XML string without an XML declaration header.
    """
    parts = ["<ClaimFeed>\n"]
    for row in sev_df.itertuples(index=False):
        parts.append(
            f"  <Claim>\n"
            f"    <PolicyId>{row.POLICY_ID}</PolicyId>\n"
            f"    <ClaimAmount>{round(row.CLAIM_AMOUNT, 2)}</ClaimAmount>\n"
            f"  </Claim>\n"
        )
    parts.append("</ClaimFeed>")
    return "".join(parts)


def load_xml_to_snowflake(
    freq_df: pd.DataFrame,
    sev_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Serialise both DataFrames to XML, upload to Snowflake, and create tables.

    Full pipeline:

    1. **Serialise** — both DataFrames are written to XML files in a local
       temp directory.  The frequency file (~678 K policies) is ~150 MB
       uncompressed; ``AUTO_COMPRESS=TRUE`` gzips it before upload.
    2. **PUT** — both files are staged in ``OUTPUT_STAGE/inbound/`` using
       the Snowflake ``PUT`` command.
       Docs: https://docs.snowflake.com/en/sql-reference/sql/put
    3. **File format** — ``XML_FF`` with ``STRIP_OUTER_ELEMENT = TRUE``
       strips the outer ``<PolicyFeed>`` / ``<ClaimFeed>`` root so each
       child element becomes a separate VARIANT row.
       Docs: https://docs.snowflake.com/en/sql-reference/sql/create-file-format
    4. **COPY INTO** — raw VARIANT staging tables ``RAW_POLICY_XML`` and
       ``RAW_CLAIM_XML`` are populated from the staged files.
       Docs: https://docs.snowflake.com/en/sql-reference/sql/copy-into-table
    5. **XMLGET parse** — pure-SQL ``CREATE TABLE … AS SELECT XMLGET(…)``
       extracts every field from the VARIANT into typed columns, producing
       ``HOME_POLICY_FREQ`` and ``HOME_POLICY_SEV`` with the same schema
       the downstream feature engineering pipeline expects.
       ``XMLGET(node, 'Tag'):"$"`` — the ``"$"`` path accessor (double-quoted)
       extracts the text content.  Nested elements use chained calls.
       Docs: https://docs.snowflake.com/en/sql-reference/functions/xmlget

    Args:
        freq_df: Transformed frequency DataFrame (~678 K rows).
        sev_df:  Transformed severity DataFrame (~26 K rows).
        args:    Parsed CLI arguments (provides database, schema).
    """
    db = args.database
    schema = args.schema
    stage = f"{db}.{schema}.STAGING"
    ff = f"{db}.{schema}.XML_FF"

    print(f"\nSerialising {len(freq_df):,} policies to XML...")
    freq_xml = generate_freq_xml(freq_df)
    print(f"  freq XML: {len(freq_xml) / 1_000_000:.1f} MB")

    print(f"Serialising {len(sev_df):,} claims to XML...")
    sev_xml = generate_sev_xml(sev_df)
    print(f"  sev  XML: {len(sev_xml) / 1_000_000:.1f} MB")

    conn = build_connection(args)
    cur = conn.cursor()

    try:
        cur.execute(f"CREATE STAGE IF NOT EXISTS {stage}")

        # Write XML strings to temp files and PUT to stage.
        # AUTO_COMPRESS=TRUE gzips the files before upload; Snowflake
        # decompresses automatically during COPY INTO.
        with tempfile.NamedTemporaryFile(
            suffix="_policy_freq.xml", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(freq_xml)
            freq_local = f.name

        with tempfile.NamedTemporaryFile(
            suffix="_policy_sev.xml", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(sev_xml)
            sev_local = f.name

        print(f"Uploading {freq_local} → @{stage}/inbound/ ...")
        cur.execute(
            f"PUT file://{freq_local} @{stage}/inbound/ "
            f"AUTO_COMPRESS=TRUE OVERWRITE=TRUE"
        )
        os.unlink(freq_local)

        print(f"Uploading {sev_local} → @{stage}/inbound/ ...")
        cur.execute(
            f"PUT file://{sev_local} @{stage}/inbound/ "
            f"AUTO_COMPRESS=TRUE OVERWRITE=TRUE"
        )
        os.unlink(sev_local)

        # XML file format — STRIP_OUTER_ELEMENT strips <PolicyFeed> /
        # <ClaimFeed> so each child element becomes a separate VARIANT row.
        cur.execute(f"""
            CREATE FILE FORMAT IF NOT EXISTS {ff}
                TYPE = XML
                STRIP_OUTER_ELEMENT = TRUE
        """)

        # ── Raw staging tables (one VARIANT row per XML element) ─────────────
        cur.execute(f"""
            CREATE OR REPLACE TABLE {db}.{schema}.RAW_POLICY_XML (SRC VARIANT)
        """)
        cur.execute(f"""
            COPY INTO {db}.{schema}.RAW_POLICY_XML
            FROM @{stage}/inbound/
            PATTERN = '.*policy_freq.*'
            FILE_FORMAT = (FORMAT_NAME = '{ff}')
            PURGE = FALSE
        """)
        rows = cur.fetchone()
        print(f"  RAW_POLICY_XML: {rows}")

        cur.execute(f"""
            CREATE OR REPLACE TABLE {db}.{schema}.RAW_CLAIM_XML (SRC VARIANT)
        """)
        cur.execute(f"""
            COPY INTO {db}.{schema}.RAW_CLAIM_XML
            FROM @{stage}/inbound/
            PATTERN = '.*policy_sev.*'
            FILE_FORMAT = (FORMAT_NAME = '{ff}')
            PURGE = FALSE
        """)
        rows = cur.fetchone()
        print(f"  RAW_CLAIM_XML:  {rows}")

        # ── HOME_POLICY_FREQ: parse VARIANT → typed columns ───────────────────
        # XMLGET(node, 'Tag'):"$" extracts the text content of the element.
        # The "$ path accessor must be double-quoted in Snowflake SQL.
        # Chained calls navigate nested elements: XMLGET(XMLGET(src,'Risk'),'X').
        cur.execute(f"""
            CREATE OR REPLACE TABLE {db}.{schema}.HOME_POLICY_FREQ AS
            SELECT
                XMLGET(SRC, 'PolicyId'):"$"::BIGINT          AS POLICY_ID,
                XMLGET(SRC, 'Exposure'):"$"::FLOAT           AS EXPOSURE,
                XMLGET(SRC, 'PolicyholderAge'):"$"::INTEGER  AS POLICYHOLDER_AGE,
                XMLGET(SRC, 'LossHistoryScore'):"$"::FLOAT   AS LOSS_HISTORY_SCORE,
                XMLGET(SRC, 'PopulationDensity'):"$"::FLOAT  AS POPULATION_DENSITY,
                XMLGET(SRC, 'RegionCode'):"$"::VARCHAR       AS REGION_CODE,
                XMLGET(XMLGET(SRC, 'Risk'), 'TerritoryCode'):"$"::VARCHAR      AS TERRITORY_CODE,
                XMLGET(XMLGET(SRC, 'Risk'), 'ConstructionType'):"$"::VARCHAR   AS CONSTRUCTION_TYPE,
                XMLGET(XMLGET(SRC, 'Risk'), 'ConstructionQuality'):"$"::INTEGER AS CONSTRUCTION_QUALITY,
                XMLGET(XMLGET(SRC, 'Risk'), 'PropertyAge'):"$"::INTEGER        AS PROPERTY_AGE,
                XMLGET(XMLGET(SRC, 'Risk'), 'OccupancyType'):"$"::VARCHAR      AS OCCUPANCY_TYPE,
                XMLGET(XMLGET(SRC, 'Claims'), 'ClaimCount'):"$"::INTEGER       AS CLAIM_COUNT
            FROM {db}.{schema}.RAW_POLICY_XML
        """)
        count = cur.execute(
            f"SELECT COUNT(*) FROM {db}.{schema}.HOME_POLICY_FREQ"
        ).fetchone()[0]
        print(f"  HOME_POLICY_FREQ: {count:,} rows")

        # ── HOME_POLICY_SEV: parse VARIANT → typed columns ────────────────────
        cur.execute(f"""
            CREATE OR REPLACE TABLE {db}.{schema}.HOME_POLICY_SEV AS
            SELECT
                XMLGET(SRC, 'PolicyId'):"$"::BIGINT    AS POLICY_ID,
                XMLGET(SRC, 'ClaimAmount'):"$"::FLOAT  AS CLAIM_AMOUNT
            FROM {db}.{schema}.RAW_CLAIM_XML
        """)
        count = cur.execute(
            f"SELECT COUNT(*) FROM {db}.{schema}.HOME_POLICY_SEV"
        ).fetchone()[0]
        print(f"  HOME_POLICY_SEV:  {count:,} rows")

        print("\nDone.")

    finally:
        cur.close()
        conn.close()


def build_connection(
    args: argparse.Namespace,
) -> snowflake.connector.SnowflakeConnection:
    """Build a Snowflake Python Connector connection from parsed CLI arguments.

    Two authentication paths are supported:

    **Named connection** (``--connection`` / ``SNOWFLAKE_CONNECTION``):
        Reads credentials from the ``[connections.<name>]`` block in
        ``~/.snowflake/connections.toml``.  No credentials appear in code or
        environment variables.  Recommended for local development.
        Docs: https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#using-a-connection-string

    **JWT / key-pair auth** (default):
        Connects using an RSA private key file.  Suitable for CI/CD pipelines
        and service accounts where interactive login is not available.
        Docs: https://docs.snowflake.com/en/user-guide/key-pair-auth

    In both cases, ``database``, ``schema``, ``role``, and ``warehouse`` are
    applied on top of the base connection so that CLI overrides take effect
    even when using a named connection with different defaults.

    Args:
        args: Parsed CLI arguments.

    Returns:
        An open ``SnowflakeConnection`` ready for cursor operations.
    """
    if args.connection:
        print(f"Connecting via named connection '{args.connection}'...")
        return snowflake.connector.connect(
            connection_name=args.connection,
            role=args.role,
            warehouse=args.warehouse,
            database=args.database,
            schema=args.schema,
        )

    print(f"Connecting to {args.account} as {args.user}...")
    return snowflake.connector.connect(
        account=args.account,
        user=args.user,
        authenticator="SNOWFLAKE_JWT",
        private_key_file=args.private_key_file,
        role=args.role,
        warehouse=args.warehouse,
        database=args.database,
        schema=args.schema,
    )


if __name__ == "__main__":
    args = parse_args()

    freq_df, sev_df = download_datasets()
    freq_df = transform_freq(freq_df)
    sev_df = transform_sev(sev_df)

    print("\nFreq column sample:")
    print(freq_df.dtypes)
    print(freq_df.head(2).to_string())

    print("\nSev column sample:")
    print(sev_df.dtypes)
    print(sev_df.head(2).to_string())

    load_xml_to_snowflake(freq_df, sev_df, args)
