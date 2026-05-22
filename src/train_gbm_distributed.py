"""
Actuarial GBM training pipeline for homeowners pure premium — distributed.

Uses ``snowflake.ml.modeling.distributors.xgboost.XGBEstimator`` to distribute
training across all nodes in the compute pool via Ray.

This module is only executable as a Snowflake ML Job — the distributors package
is not available outside the Container Runtime.  Submit the entire src/ directory
so that config.py and helpers.py are available as imports inside the container.

Usage
-----
As a Snowflake Multi-Node ML Job (from submit_job.py or equivalent)::

    from snowflake.ml.jobs import submit_directory

    job = submit_directory(
        "src/",
        entrypoint="train_gbm_distributed.py",
        compute_pool="DEMO_POOL",
        stage_name="payload_stage",
        query_warehouse="COMPUTE_WH",
        target_instances=3,             # 1 head + 2 workers
        args=["--ds-version=1"],
        session=session,
    )
    job.wait()
    print(job.status)

Snowflake features used
-----------------------
Multi-Node ML Jobs:
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/ml-jobs/distributed-ml-jobs
Distributed XGBoost:
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/distributed-training
Feature Store (datasets):
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview
Experiment Tracking:
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiment-tracking
Model Registry:
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before any other plt import
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.dataset import load_dataset
from snowflake.ml.experiment import ExperimentTracking
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session

from config import DATABASE, SCHEMA
from helpers import double_lift_chart, lorenz_curve
try:
        from snowflake.ml.modeling.distributors.xgboost import (  # type: ignore
        XGBEstimator,
        XGBScalingConfig,
    )
except ImportError as err:
    print("Package only available within SPCS service. This script is intended only to run as an MLJob")
    raise(err)


# Spine columns added by the Feature Store at dataset generation time.
# These are excluded when deriving the feature column list for training.
_SPINE_COLS = {"POLICY_ID", "PURE_PREMIUM", "EXPOSURE"}

# Snowflake Experiments groups runs into a named experiment so all training
# runs for this model are visible together in the Snowsight Experiments UI.
EXPERIMENT = "ACTUARIAL_GBM_TRAINING"



def train(
    session: Session,
    ds_version: str = "1",
    model_version: str = "v1",
    n_estimators: int = 200,
) -> None:
    """Run the end-to-end actuarial distributed GBM training pipeline.

    Steps performed:

    1. **Experiment Tracking setup** — creates or resumes a named experiment
       and opens a run via context manager.  All params, metrics, artifacts,
       and the model version are linked to this run in the Snowsight UI.
    2. **Feature Store dataset load** — reads the pre-built, versioned
       ACTUARIAL_TRAINING / ACTUARIAL_VALIDATION datasets.  Training data is
       wrapped in a ``DataConnector`` which reads Parquet files from the
       internal stage directly via Ray — no warehouse SQL, no data egress to
       the local machine.
    3. **Hyperparameter logging** — stores all training parameters as
       key-value pairs visible in the run comparison view.
    4. **Distributed XGBoost training** — ``XGBEstimator`` trains across all
       nodes in the compute pool via Ray.  ``XGBScalingConfig()`` with defaults
       (``num_workers=-1``) automatically assigns one worker per node and
       allocates all available CPUs per worker.
    5. **Evaluation** — RMSE and MAE computed from pandas predictions on the
       validation set (pulled once; reused for plots in step 7).
    6. **Model registration** — ``tracker.log_model`` registers the native
       ``xgboost.Booster`` (extracted via ``get_booster()``) in the Snowflake
       Model Registry and links the version to this experiment run.
    7. **Artifact upload** — diagnostic plots saved to ``/tmp`` and uploaded
       to the experiment's artifact store; visible in the Artifacts tab.

    Args:
        session:       Active Snowpark session (injected by Container Runtime).
        ds_version:    Version tag of the ACTUARIAL_TRAINING and
                       ACTUARIAL_VALIDATION Feature Store datasets to use.
                       Increment this when the feature engineering pipeline
                       (ML_INPUT_SNOWFLAKE) has been re-run.
        model_version: Label for this run in the Experiment Tracking UI and
                       in the Model Registry.  Defaults to the
                       ``SNOWFLAKE_SERVICE_NAME`` env var when run as an
                       ML Job (auto-unique per submission).
        n_estimators:  Number of boosting rounds for XGBoost.  Increase for
                       potentially better accuracy at the cost of longer
                       training time.  Logged as a hyperparameter so runs
                       with different values are comparable in Snowsight.

    Note:
        Do NOT call ``session.use_warehouse()`` or ``USE ROLE`` inside this
        function.  When running as an ML Job the warehouse is configured at
        submission time via ``query_warehouse``; the role is fixed by the OAuth
        token scope.  Both are blocked inside the container.
    """
    # Set database and schema from config.  Role is fixed by the OAuth token
    # issued at job submission time — USE ROLE is blocked inside the container.
    print(f"{DATABASE=}")
    print(f"{SCHEMA=}")
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)

    # ── 1. Experiment Tracking setup ─────────────────────────────────────────
    # ``start_run`` returns a context manager — the run is automatically
    # closed (and committed) on exit, even if an exception is raised.
    # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiment-tracking
    tracker = ExperimentTracking(
        session=session,
        database_name=DATABASE,
        schema_name=SCHEMA,
    )
    tracker.set_experiment(EXPERIMENT)

    with tracker.start_run(run_name=model_version):
        # ── 2. Load pre-built Feature Store datasets ──────────────────────────
        # ``load_dataset`` reads a versioned dataset that was generated by the
        # Feature Store (``fs.generate_dataset``).  The dataset is stored as a
        # materialised Parquet snapshot in Snowflake — no feature recomputation
        # happens here.  Versioning ensures the exact same rows and columns used
        # for this training run can be reproduced later for audit purposes.
        # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview
        training_dataset = load_dataset(
            session, name="ACTUARIAL_TRAINING", version=ds_version
        )
        validation_dataset = load_dataset(
            session, name="ACTUARIAL_VALIDATION", version=ds_version
        )

        # Column names only — lazy plan inspection, no SQL executed.
        feature_cols = [
            c for c in training_dataset.read.to_snowpark_dataframe().columns
            if c not in _SPINE_COLS
        ]
        print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]} ...")

        # DataConnector.from_dataset reads Parquet files from the internal stage
        # directly via Ray — no warehouse SQL, no multi-query plan issue.
        train_connector = DataConnector.from_dataset(training_dataset)
        val_sdf = validation_dataset.read.to_snowpark_dataframe()

        # ── 3. Log hyperparameters ────────────────────────────────────────────
        # ``log_params`` stores key-value pairs in the run record.  These are
        # displayed in the Parameters tab in Snowsight and are included in the
        # run comparison table when evaluating multiple experiments.
        hparams = dict(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_leaves=31,
            objective="reg:squarederror",
            ds_version=ds_version,
        )
        tracker.log_params(hparams)

        # ── 4. Train distributed XGBoost ──────────────────────────────────────
        # ``XGBEstimator`` distributes training across all Ray workers in the
        # compute pool.  ``XGBScalingConfig()`` with defaults (num_workers=-1,
        # num_cpu_per_worker=-1) lets the runtime assign one worker per node
        # and all available CPUs per worker automatically.
        # ``DataConnector.from_dataset`` reads Parquet from the internal stage
        # directly — no warehouse SQL, no multi-query plan constraint.
        # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/distributed-training
        gbm = XGBEstimator(
            n_estimators=n_estimators,
            objective="reg:squarederror",
            params={
                "learning_rate": 0.05,
                "max_leaves": 31,
            },
            scaling_config=XGBScalingConfig(),
        )
        booster = gbm.fit(train_connector, input_cols=feature_cols, label_col="PURE_PREMIUM")
        print("Model training complete")

        # ── 5. Evaluate ───────────────────────────────────────────────────────
        # Pull the validation set to pandas once; predictions are computed
        # locally from the fitted model and the same df_val is reused for
        # plots in step 7, avoiding a second round-trip.
        df_val = val_sdf.to_pandas()
        df_val["PREDICTED_PURE_PREMIUM"] = gbm.predict(df_val[feature_cols].values)

        val_rmse = float(
            np.sqrt(np.mean((df_val["PURE_PREMIUM"] - df_val["PREDICTED_PURE_PREMIUM"]) ** 2))
        )
        val_mae = float(
            np.mean(np.abs(df_val["PURE_PREMIUM"] - df_val["PREDICTED_PURE_PREMIUM"]))
        )
        print(f"Validation RMSE : {val_rmse:>10.4f}")
        print(f"Validation MAE  : {val_mae:>10.4f}")

        # ``log_metrics`` stores numeric values in the run record.  Use the
        # ``step`` parameter (default 0) for time-series metrics such as
        # per-epoch training loss.
        tracker.log_metrics({"val_rmse": val_rmse, "val_mae": val_mae})

        # ── 6. Log model ──────────────────────────────────────────────────────
        # ``tracker.log_model`` does two things in one call:
        #   (a) registers the fitted model as a versioned entry in the Snowflake
        #       Model Registry (visible under AI & ML → Models in Snowsight), and
        #   (b) links that model version back to this experiment run so the run
        #       record shows which model it produced.
        # ``target_platforms`` controls where the model can be served:
        #   WAREHOUSE  → mv.run() for batch scoring on a virtual warehouse
        #   SNOWPARK_CONTAINER_SERVICES → mv.run_batch() on a compute pool
        # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview
        # XGBEstimator itself is not a type the Model Registry packager supports.
        # Extract the underlying xgboost.Booster via get_booster() before logging.
        tracker.log_model(
            model=booster,
            model_name="ACTUARIAL_GBM",
            version_name=model_version,
            comment="Distributed XGBoost GBM for homeowners pure premium pricing.",
            metrics={"val_rmse": val_rmse, "val_mae": val_mae},
            options={"enable_explainability": True},
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            sample_input_data=training_dataset.read.to_snowpark_dataframe().select(feature_cols).limit(10),
        )  # type: ignore

        # ── 7. Generate and upload diagnostic plots ───────────────────────────
        # Plots are saved to /tmp (available in both local and SPCS container
        # environments) then uploaded via ``log_artifact``.  Artifacts are
        # stored in the experiment's internal Snowflake stage and appear in the
        # Artifacts tab of the run in Snowsight.
        # df_val and PREDICTED_PURE_PREMIUM already populated in step 5.
        df_val["PurePremium"] = df_val["PURE_PREMIUM"]

        # double_lift_chart creates and returns its own figure — no plt.subplots()
        # call is needed before it.
        fig_lift = double_lift_chart(
            df_val, {"GBM": df_val["PREDICTED_PURE_PREMIUM"].values}, n_bins=10
        )
        fig_lift.savefig("/tmp/double_lift.png", dpi=150, bbox_inches="tight")
        plt.close(fig_lift)

        fig_gini, ax = plt.subplots(figsize=(8, 8))
        cum_exp, cum_claims = lorenz_curve(
            df_val["PURE_PREMIUM"], df_val["PREDICTED_PURE_PREMIUM"], df_val["EXPOSURE"]
        )
        gini = 1 - 2 * auc(cum_exp, cum_claims)
        ax.plot(cum_exp, cum_claims, label=f"GBM (Gini={gini:.3f})")
        ax.plot([0, 1], [0, 1], "--k", label="Random")
        ax.legend()
        fig_gini.savefig("/tmp/gini_lorenz.png", dpi=150, bbox_inches="tight")
        plt.close(fig_gini)

        tracker.log_artifact("/tmp/double_lift.png")
        tracker.log_artifact("/tmp/gini_lorenz.png")

if __name__ == "__main__":
    # SNOWFLAKE_SERVICE_NAME is injected automatically into the container
    # environment by the SPCS runtime.  Using it as the default model version
    # means every job submission produces a uniquely named model version
    # (e.g. "TRAIN_GBM_DISTRIBUTED_8CA1356D_VFNK33GC6RQD") with zero manual
    # bookkeeping.  The version is also the experiment run name, so the model
    # version and the run that produced it share the same ID.
    default_version = os.environ.get("SNOWFLAKE_SERVICE_NAME", "v1")

    parser = argparse.ArgumentParser(description="Train actuarial GBM model (distributed).")
    parser.add_argument(
        "--ds-version", default="1", help="Feature Store dataset version"
    )
    parser.add_argument(
        "--model-version",
        default=default_version,
        help="Experiment run name / Model Registry version",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        dest="n_estimators",
        help="Number of XGBoost boosting rounds (default: 200).",
    )
    args = parser.parse_args()

    session = get_active_session()
    train(
        session,
        ds_version=args.ds_version,
        model_version=args.model_version,
        n_estimators=args.n_estimators,
    )
