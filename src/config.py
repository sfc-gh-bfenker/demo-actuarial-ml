"""
Central configuration for the actuarial ML demo.

These defaults match setup.sql exactly — a fresh setup.sql run requires no edits
here. To customize object names, see CUSTOMIZING.md.

Docs:
    Snowflake named connections:
    https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#using-a-connection-string
"""

# ── Snowflake environment ─────────────────────────────────────────────────────
DATABASE = "ACTUARIAL_DEMO_DB"  # Target database
SCHEMA = "ACTUARIAL_PRICING"  # Target schema
ROLE = "ACTUARIAL_DEMO_ROLE"  # Role used for all operations
WAREHOUSE = "ACTUARIAL_DEMO_WH"  # Virtual warehouse for SQL compute
COMPUTE_POOL = "ACTUARIAL_DEMO_POOL"  # Compute pool for ML Jobs and batch inference
STAGE = "payload_stage"  # Stage name for ML Job payload uploads


def create_session():
    """Return a Snowpark session for local development or container execution.

    When running inside a Snowflake container (ML Job, Notebook on Container
    Runtime), ``get_active_session()`` returns the pre-configured session with
    no credentials required — Snowflake injects the OAuth token automatically.

    For local development the fallback reads connection details from the
    ``default`` named connection in ``~/.snowflake/connections.toml`` using
    the constants defined in this file.

    Returns:
        Active ``snowflake.snowpark.Session``.
    """
    try:
        from snowflake.snowpark.context import get_active_session

        return get_active_session()
    except Exception:
        from snowflake.snowpark import Session

        return Session.builder.configs(
            {
                "connection_name": "default",
                "role": ROLE,
                "warehouse": WAREHOUSE,
                "database": DATABASE,
                "schema": SCHEMA,
            }
        ).create()
