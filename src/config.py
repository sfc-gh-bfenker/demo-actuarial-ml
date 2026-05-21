"""
Central configuration for the actuarial ML demo.

Edit this file to match your Snowflake environment before running
``load_actuarial_data.py`` or the notebook.

NOTE: ``train.py`` runs as a standalone file inside a Snowflake ML Job
container and cannot import from this file.  Update its constants
(DATABASE, SCHEMA, ROLE, WAREHOUSE) directly when changing environments.

Docs:
    Snowflake named connections:
    https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#using-a-connection-string
"""

# ── Snowflake environment ─────────────────────────────────────────────────────
DATABASE = "COUNTRY_ML"  # Target database
SCHEMA = "ACTUARIAL_PRICING"  # Target schema
ROLE = "ACCOUNTADMIN"  # Role used for all operations
WAREHOUSE = "COMPUTE_WH"  # Virtual warehouse for SQL compute
COMPUTE_POOL = "CPU_POOL"  # Compute pool for ML Jobs and batch inference

# ── load_actuarial_data.py only ───────────────────────────────────────────────
# These default to empty strings.  Override via CLI flags or environment
# variables (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_FILE).
DEFAULT_ACCOUNT = ""  # e.g. "myorg-myaccount"
DEFAULT_USER = ""  # e.g. "myuser"
DEFAULT_PRIVATE_KEY_FILE = ""  # e.g. "/home/user/.snowflake/rsa_key.p8"
