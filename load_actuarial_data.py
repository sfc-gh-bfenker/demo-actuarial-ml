"""
Load freMTPL2 (French Motor TPL) dataset into Snowflake as a homeowners insurance dataset.

The `freMTPL2 <https://www.openml.org/d/41214>`_ dataset is a French motor
third-party liability portfolio (~678 K policies) published on OpenML.  Column
names and value domains are remapped here to match a homeowners insurance
context so the demo is directly relatable to property/casualty actuaries.

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
      --xml-sample-size   number of policies to serialize as XML (default 1000)
      --skip-xml          skip XML sample generation and loading

Tables written
--------------
``<database>.<schema>.HOME_POLICY_FREQ``
    ~678 K rows — policy-level frequency data (one row per policy).

``<database>.<schema>.HOME_POLICY_SEV``
    ~26 K rows — claim-level severity data (one row per claim; multiple rows
    per policy are possible).

``<database>.<schema>.RAW_POLICY_XML``
    ``xml_sample_size`` rows — raw VARIANT column; one row per ``<Policy>``
    element loaded from the XML file via ``COPY INTO`` with
    ``STRIP_OUTER_ELEMENT = TRUE``.

``<database>.<schema>.HOME_POLICY_FREQ_XML``
    ``xml_sample_size`` rows — parsed from ``RAW_POLICY_XML`` using
    ``XMLGET``; same column schema as ``HOME_POLICY_FREQ``.  Demonstrates the
    Ratabase / PolicyPro XML ingestion pattern end-to-end.

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
import xml.etree.ElementTree as ET

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ── SSL patch (corporate certificate environment) ─────────────────────────────
# Some corporate networks intercept HTTPS with a custom CA.  Disabling
# verification allows ``fetch_openml`` to reach openml.org.  Remove this line
# if your network does not require it.
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_openml  # noqa: E402 (import after ssl patch)

# ── Connection defaults ───────────────────────────────────────────────────────
# Override any of these via the corresponding CLI flag or environment variable.
DEFAULT_ACCOUNT = "SFSENORTHAMERICA-BFENKER_AWS1"
DEFAULT_USER = "BFENKER"
DEFAULT_ROLE = "COUNTRY_BANK_DEMO_ROLE"
DEFAULT_WAREHOUSE = "COMPUTE_WH"
DEFAULT_DATABASE = "COUNTRY_BANK_DEMO_DB"
DEFAULT_SCHEMA = "ACTUARIAL_PRICING"
DEFAULT_PRIVATE_KEY_FILE = "/Users/bfenker/.snowflake/rsa_key.p8"

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
    parser.add_argument(
        "--xml-sample-size",
        type=int,
        default=1000,
        dest="xml_sample_size",
        help="Number of policies to serialize as XML for the ingestion demo (default: 1000).",
    )
    parser.add_argument(
        "--skip-xml",
        action="store_true",
        dest="skip_xml",
        help="Skip XML sample generation and Snowflake loading.",
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
        Cleaned and renamed DataFrame ready for ``load_to_snowflake()``.
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

    # Explicit dtype coercion ensures write_pandas infers Snowflake column
    # types correctly (e.g. BIGINT vs FLOAT) instead of relying on pandas
    # defaults which can vary across pandas versions.
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


def generate_xml_sample(freq_df: pd.DataFrame, n: int = 1000) -> str:
    """Serialize a random sample of HOME_POLICY_FREQ rows to a PolicyFeed XML string.

    The XML schema mimics a Ratabase / PolicyPro-style property rating extract,
    with nested ``<Risk>`` and ``<Claims>`` sub-elements to demonstrate
    Snowflake's ability to parse nested XML structures using chained
    ``XMLGET`` calls.

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

    The outer ``<PolicyFeed>`` root is stripped by the Snowflake XML file
    format option ``STRIP_OUTER_ELEMENT = TRUE``, which causes each direct
    child ``<Policy>`` element to be loaded as a separate VARIANT row in
    ``RAW_POLICY_XML``.  This is the recommended pattern for loading
    multi-record XML files into Snowflake.

    Docs:
        https://docs.snowflake.com/en/sql-reference/sql/create-file-format
        (see ``STRIP_OUTER_ELEMENT`` parameter)

    Args:
        freq_df: Transformed frequency DataFrame (output of ``transform_freq``).
        n:       Number of policies to include.  ``random_state=42`` is used
                 for reproducible samples across runs.

    Returns:
        UTF-8 XML string (no XML declaration header; the file format handles
        encoding at load time).
    """
    sample = freq_df.sample(min(n, len(freq_df)), random_state=42)

    root = ET.Element("PolicyFeed")
    for _, row in sample.iterrows():
        policy = ET.SubElement(root, "Policy")
        ET.SubElement(policy, "PolicyId").text = str(int(row["POLICY_ID"]))
        ET.SubElement(policy, "Exposure").text = str(round(float(row["EXPOSURE"]), 6))
        ET.SubElement(policy, "PolicyholderAge").text = str(
            int(row["POLICYHOLDER_AGE"])
        )
        ET.SubElement(policy, "LossHistoryScore").text = str(
            round(float(row["LOSS_HISTORY_SCORE"]), 4)
        )
        ET.SubElement(policy, "PopulationDensity").text = str(
            round(float(row["POPULATION_DENSITY"]), 4)
        )
        ET.SubElement(policy, "RegionCode").text = str(row["REGION_CODE"])

        risk = ET.SubElement(policy, "Risk")
        ET.SubElement(risk, "TerritoryCode").text = str(row["TERRITORY_CODE"])
        ET.SubElement(risk, "ConstructionType").text = str(row["CONSTRUCTION_TYPE"])
        ET.SubElement(risk, "ConstructionQuality").text = str(
            int(row["CONSTRUCTION_QUALITY"])
        )
        ET.SubElement(risk, "PropertyAge").text = str(int(row["PROPERTY_AGE"]))
        ET.SubElement(risk, "OccupancyType").text = str(row["OCCUPANCY_TYPE"])

        claims = ET.SubElement(policy, "Claims")
        ET.SubElement(claims, "ClaimCount").text = str(int(row["CLAIM_COUNT"]))

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def load_xml_to_snowflake(freq_df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Generate a PolicyFeed XML sample and load it into Snowflake.

    This function demonstrates the full XML ingestion pipeline that mirrors
    how a carrier would receive a Ratabase or PolicyPro rating extract:

    1. **Generate** — serialize a sample of ``HOME_POLICY_FREQ`` rows to a
       well-structured ``<PolicyFeed>`` XML document.
    2. **PUT** — upload the local XML file to ``OUTPUT_STAGE`` using the
       Snowflake Python Connector's ``PUT`` command, which stages files for
       bulk loading without requiring external storage access.
       Docs: https://docs.snowflake.com/en/sql-reference/sql/put
    3. **FILE FORMAT** — create an XML file format with
       ``STRIP_OUTER_ELEMENT = TRUE``, which strips the ``<PolicyFeed>`` root
       and loads each ``<Policy>`` child as a separate row.
       Docs: https://docs.snowflake.com/en/sql-reference/sql/create-file-format
    4. **COPY INTO** — load the staged XML into ``RAW_POLICY_XML`` as a
       ``VARIANT`` column.  The VARIANT type stores the XML element as a
       native semi-structured value that Snowflake can query with path
       expressions.
       Docs: https://docs.snowflake.com/en/sql-reference/sql/copy-into-table
    5. **XMLGET parse** — create ``HOME_POLICY_FREQ_XML`` by extracting each
       field from the VARIANT using ``XMLGET``.  Key syntax notes:

       - ``XMLGET(src, 'Tag'):$`` — the ``:$`` path extracts the text content
         of the element.
       - ``XMLGET(XMLGET(src, 'Risk'), 'TerritoryCode'):$`` — chained calls
         navigate nested elements.
       - Cast with ``::TYPE`` to produce typed columns.
       Docs: https://docs.snowflake.com/en/sql-reference/functions/xmlget

    Args:
        freq_df: Transformed frequency DataFrame.
        args:    Parsed CLI arguments (provides database, schema, xml_sample_size).
    """
    db = args.database
    schema = args.schema
    stage = f"{db}.{schema}.OUTPUT_STAGE"
    ff = f"{db}.{schema}.XML_FF"

    print(f"\nGenerating XML sample ({args.xml_sample_size} policies)...")
    xml_str = generate_xml_sample(freq_df, args.xml_sample_size)

    conn = build_connection(args)
    cur = conn.cursor()

    try:
        # Ensure the output stage exists before attempting to PUT files.
        cur.execute(f"CREATE STAGE IF NOT EXISTS {stage}")

        # Write the XML string to a temporary local file, then PUT it to the
        # Snowflake stage.  AUTO_COMPRESS=FALSE keeps the file as plain XML so
        # it is readable in the Snowsight Files view; Snowflake handles gzip
        # transparently if AUTO_COMPRESS=TRUE were used instead.
        with tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(xml_str)
            local_path = f.name

        print(f"Uploading {local_path} → @{stage}/inbound/ ...")
        cur.execute(
            f"PUT file://{local_path} @{stage}/inbound/ "
            f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        )
        os.unlink(local_path)

        # STRIP_OUTER_ELEMENT = TRUE removes the <PolicyFeed> root so that
        # each <Policy> child element becomes a separate row in the target table,
        # rather than loading the entire document as a single VARIANT.
        cur.execute(f"""
            CREATE FILE FORMAT IF NOT EXISTS {ff}
                TYPE = XML
                STRIP_OUTER_ELEMENT = TRUE
        """)

        # RAW_POLICY_XML stores each <Policy> element verbatim as a VARIANT.
        # This preserves the original XML structure for auditability and allows
        # ad-hoc querying with XMLGET without committing to a fixed schema
        # upfront — useful when the source format may evolve.
        cur.execute(f"""
            CREATE OR REPLACE TABLE {db}.{schema}.RAW_POLICY_XML (SRC VARIANT)
        """)
        cur.execute(f"""
            COPY INTO {db}.{schema}.RAW_POLICY_XML
            FROM @{stage}/inbound/
            FILE_FORMAT = (FORMAT_NAME = '{ff}')
            PURGE = FALSE
        """)
        rows_loaded = cur.fetchone()
        print(f"  RAW_POLICY_XML loaded: {rows_loaded}")

        # Parse the VARIANT XML into typed columns matching HOME_POLICY_FREQ.
        # XMLGET(node, 'Tag'):$ extracts the text content of the named element.
        # Chained calls like XMLGET(XMLGET(src, 'Risk'), 'TerritoryCode')
        # navigate nested elements without any schema pre-declaration.
        cur.execute(f"""
            CREATE OR REPLACE TABLE {db}.{schema}.HOME_POLICY_FREQ_XML AS
            SELECT
                XMLGET(SRC, 'PolicyId'):$::BIGINT          AS POLICY_ID,
                XMLGET(SRC, 'Exposure'):$::FLOAT           AS EXPOSURE,
                XMLGET(SRC, 'PolicyholderAge'):$::INTEGER  AS POLICYHOLDER_AGE,
                XMLGET(SRC, 'LossHistoryScore'):$::FLOAT   AS LOSS_HISTORY_SCORE,
                XMLGET(SRC, 'PopulationDensity'):$::FLOAT  AS POPULATION_DENSITY,
                XMLGET(SRC, 'RegionCode'):$::VARCHAR       AS REGION_CODE,
                XMLGET(XMLGET(SRC, 'Risk'), 'TerritoryCode'):$::VARCHAR      AS TERRITORY_CODE,
                XMLGET(XMLGET(SRC, 'Risk'), 'ConstructionType'):$::VARCHAR   AS CONSTRUCTION_TYPE,
                XMLGET(XMLGET(SRC, 'Risk'), 'ConstructionQuality'):$::INTEGER AS CONSTRUCTION_QUALITY,
                XMLGET(XMLGET(SRC, 'Risk'), 'PropertyAge'):$::INTEGER        AS PROPERTY_AGE,
                XMLGET(XMLGET(SRC, 'Risk'), 'OccupancyType'):$::VARCHAR      AS OCCUPANCY_TYPE,
                XMLGET(XMLGET(SRC, 'Claims'), 'ClaimCount'):$::INTEGER       AS CLAIM_COUNT
            FROM {db}.{schema}.RAW_POLICY_XML
        """)
        print(
            f"  HOME_POLICY_FREQ_XML created — same schema as HOME_POLICY_FREQ, sourced from XML."
        )

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


def load_to_snowflake(
    freq_df: pd.DataFrame,
    sev_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Load the frequency and severity DataFrames into Snowflake tables.

    Uses ``write_pandas`` from the Snowflake Python Connector, which:

    - Stages the DataFrame as compressed Parquet files in a temporary stage.
    - Issues a ``COPY INTO`` to load from the stage into the target table.
    - With ``auto_create_table=True`` the table DDL is inferred from the
      DataFrame's dtypes (``int64`` → ``BIGINT``, ``float64`` → ``FLOAT``,
      ``object`` → ``VARCHAR``).
    - With ``overwrite=True`` the table is truncated before each load, making
      re-runs idempotent.

    Docs:
        https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-pandas

    Args:
        freq_df: Transformed HOME_POLICY_FREQ DataFrame (~678 K rows).
        sev_df:  Transformed HOME_POLICY_SEV DataFrame (~26 K rows).
        args:    Parsed CLI arguments (provides database, schema).
    """
    conn = build_connection(args)

    print(f"Loading HOME_POLICY_FREQ into {args.database}.{args.schema}...")
    success, nchunks, nrows, _ = write_pandas(
        conn=conn,
        df=freq_df,
        table_name="HOME_POLICY_FREQ",
        database=args.database,
        schema=args.schema,
        auto_create_table=True,
        overwrite=True,
        quote_identifiers=False,
    )
    print(f"  HOME_POLICY_FREQ: success={success}, chunks={nchunks}, rows={nrows}")

    print(f"Loading HOME_POLICY_SEV into {args.database}.{args.schema}...")
    success, nchunks, nrows, _ = write_pandas(
        conn=conn,
        df=sev_df,
        table_name="HOME_POLICY_SEV",
        database=args.database,
        schema=args.schema,
        auto_create_table=True,
        overwrite=True,
        quote_identifiers=False,
    )
    print(f"  HOME_POLICY_SEV:  success={success}, chunks={nchunks}, rows={nrows}")

    conn.close()
    print("\nDone.")


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

    load_to_snowflake(freq_df, sev_df, args)

    if not args.skip_xml:
        load_xml_to_snowflake(freq_df, args)
