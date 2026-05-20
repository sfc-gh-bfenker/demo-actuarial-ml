"""
Load freMTPL2 (French Motor TPL) dataset into Snowflake as a homeowners insurance dataset.

Usage:
    python load_actuarial_data.py [OPTIONS]

    All connection/target parameters default to the original demo values but can
    be overridden via CLI flags or environment variables:

      --connection  SNOWFLAKE_CONNECTION  named connection from connections.toml
      --account     SNOWFLAKE_ACCOUNT
      --user        SNOWFLAKE_USER
      --role        SNOWFLAKE_ROLE
      --warehouse   SNOWFLAKE_WAREHOUSE
      --database    SNOWFLAKE_DATABASE
      --schema      SNOWFLAKE_SCHEMA
      --private-key-file  SNOWFLAKE_PRIVATE_KEY_FILE
      --xml-sample-size   Number of policies to serialize as XML (default 1000)
      --skip-xml          Skip XML sample generation and loading

Tables written:
    <database>.<schema>.HOME_POLICY_FREQ      (~678K rows) - policy-level frequency data
    <database>.<schema>.HOME_POLICY_SEV       (~26K rows)  - claim-level severity data
    <database>.<schema>.RAW_POLICY_XML        (xml_sample_size rows) - raw VARIANT XML
    <database>.<schema>.HOME_POLICY_FREQ_XML  (xml_sample_size rows) - parsed from XML,
                                               same schema as HOME_POLICY_FREQ
"""

import argparse
import os
import ssl
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ── SSL patch (corporate cert environment) ────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_openml  # noqa: E402 (import after ssl patch)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ACCOUNT = "SFSENORTHAMERICA-BFENKER_AWS1"
DEFAULT_USER = "BFENKER"
DEFAULT_ROLE = "COUNTRY_BANK_DEMO_ROLE"
DEFAULT_WAREHOUSE = "COMPUTE_WH"
DEFAULT_DATABASE = "COUNTRY_BANK_DEMO_DB"
DEFAULT_SCHEMA = "ACTUARIAL_PRICING"
DEFAULT_PRIVATE_KEY_FILE = "/Users/bfenker/.snowflake/rsa_key.p8"

# ── Column rename maps ────────────────────────────────────────────────────────
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

# ISO construction class lookup (maps freMTPL2 VehBrand codes → homeowners classes)
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

OCCUPANCY_TYPE_MAP = {
    "Regular": "Owner-Occupied",
    "Diesel": "Tenant-Occupied",
}


def parse_args() -> argparse.Namespace:
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
    df = df.rename(columns=FREQ_RENAME)

    # Value remaps — strip ARFF-injected single quotes then map
    # (VehGas arrives as object dtype with values like "'Regular'"; VehBrand as categorical, already clean)
    df["OCCUPANCY_TYPE"] = (
        df["OCCUPANCY_TYPE"].astype(str).str.strip("'").map(OCCUPANCY_TYPE_MAP)
    )
    df["CONSTRUCTION_TYPE"] = (
        df["CONSTRUCTION_TYPE"].astype(str).str.strip("'").map(CONSTRUCTION_TYPE_MAP)
    )

    # Ensure numeric types are correct
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
    df = df.rename(columns=SEV_RENAME)
    df["POLICY_ID"] = df["POLICY_ID"].astype("int64")
    df["CLAIM_AMOUNT"] = df["CLAIM_AMOUNT"].astype("float64")
    return df


def generate_xml_sample(freq_df: pd.DataFrame, n: int = 1000) -> str:
    """Serialize a sample of HOME_POLICY_FREQ to a PolicyFeed XML string.

    The structure mimics a Ratabase/PolicyPro-style property rating extract:

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
    """Generate a PolicyFeed XML sample, upload it to OUTPUT_STAGE, and load it into
    Snowflake as RAW_POLICY_XML (VARIANT) and HOME_POLICY_FREQ_XML (parsed columns).

    Demonstrates the Ratabase/PolicyPro XML ingestion pattern:
        PUT → COPY INTO RAW_POLICY_XML → XMLGET() → HOME_POLICY_FREQ_XML
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
        # Ensure OUTPUT_STAGE exists
        cur.execute(f"CREATE STAGE IF NOT EXISTS {stage}")

        # Write XML to a temp file and PUT it to the stage
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

        # XML file format — STRIP_OUTER_ELEMENT splits <PolicyFeed> children into rows
        cur.execute(f"""
            CREATE FILE FORMAT IF NOT EXISTS {ff}
                TYPE = XML
                STRIP_OUTER_ELEMENT = TRUE
        """)

        # Raw staging table: one VARIANT row per <Policy> element
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

        # Parsed table: XMLGET extracts each field into typed columns.
        # XMLGET(node, 'Tag'):$ is the text content of the element.
        # Nested elements use chained XMLGET calls.
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
    """Build a Snowflake connection from parsed args.

    If --connection is provided, use a named connection from connections.toml
    (only database/schema/role/warehouse are overridden if also supplied).
    Otherwise, connect with explicit credentials using JWT auth.
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
):
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
