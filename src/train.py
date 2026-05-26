"""
Actuarial GBM training pipeline for homeowners pure premium.

Reads pre-built ACTUARIAL_TRAINING and ACTUARIAL_VALIDATION datasets from the
Snowflake Feature Store, trains a Snowflake-managed XGBoost model, tracks the
run in Snowflake Experiments, and registers the model in the Model Registry.

Usage
-----
Local::

    python train.py
    python train.py --ds-version 2 --model-version v2

As a Snowflake ML Job::

    from snowflake.ml.jobs import submit_directory

    job = submit_directory(
        "src/",
        entrypoint="train.py",
        compute_pool="DEMO_POOL",
        stage_name="payload_stage",
        query_warehouse="COMPUTE_WH",
        args=["--ds-version=1"],
        session=session,
    )
    job.wait()
    print(job.status)

Snowflake features used
-----------------------
ML Jobs:
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/ml-jobs/overview
Feature Store (datasets):
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview
Experiment Tracking:
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiment-tracking
Model Registry:
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview
Snowpark ML Modeling (XGBRegressor):
    https://docs.snowflake.com/en/developer-guide/snowflake-ml/modeling/overview
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before any other plt import

from snowflake.ml.dataset import load_dataset
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.modeling.xgboost import (
    XGBRegressor,  # type: ignore -> available on SPCS
)
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F

from config import DATABASE, SCHEMA, create_session
from helpers import generate_diagnostic_plots

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
    """Run the end-to-end actuarial GBM training pipeline.

    Steps performed:

    1. **Experiment Tracking setup** — creates or resumes a named experiment
       and opens a run.  All params, metrics, artifacts, and the model version
       are linked to this run in the Snowsight Experiments UI.
    2. **Feature Store dataset load** — reads the pre-built, versioned
       ACTUARIAL_TRAINING / ACTUARIAL_VALIDATION datasets directly from
       Snowflake without any data egress to the local machine.
    3. **Hyperparameter logging** — stores all training parameters as
       key-value pairs visible in the run comparison view.
    4. **Snowflake-managed XGBoost training** — ``XGBRegressor`` from
       ``snowflake.ml.modeling`` wraps XGBoost with a Snowpark DataFrame
       interface; training executes on Snowflake compute (warehouse or
       compute pool depending on call site).
    5. **Server-side evaluation** — RMSE and MAE are computed as Snowpark SQL
       aggregations; no rows are transferred to the client for scoring.
    6. **Generate diagnostic plots** — double-lift and Gini/Lorenz charts are
       produced and saved to ``/tmp`` for attachment to the model version.
    7. **Log model** — ``tracker.log_model`` registers the fitted model in
       the Snowflake Model Registry AND links the version to this experiment
       run in a single call.  Diagnostic plots are uploaded as artifacts.

    Args:
        session:       Active Snowpark session.
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
        Do NOT call ``session.use_warehouse()`` or ``session.sql("USE ROLE ...")``
        inside this function.  When running as an ML Job the warehouse and role
        are configured at submission time via the ``query_warehouse`` argument to
        ``submit_directory``; issuing ``USE WAREHOUSE`` or ``USE ROLE`` from
        inside the container is rejected by Snowflake.
    """
    # Set database and schema explicitly so the function is portable
    # across execution environments (local, ML Job, Snowflake Notebook).
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)

    # ── 1. Experiment Tracking setup ─────────────────────────────────────────
    # ExperimentTracking is an MLflow-compatible singleton.  ``set_experiment``
    # creates the experiment object in Snowflake if it does not already exist.
    # ``start_run`` opens a run context; the context manager closes it and
    # persists all logged data.
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

        # ``to_snowpark_dataframe()`` returns a lazy Snowpark DataFrame — no
        # data is transferred until an action (collect, to_pandas, fit) is
        # called.  Pass ``only_feature_cols=True`` at inference time to drop
        # the label columns automatically.
        train_sdf = training_dataset.read.to_snowpark_dataframe()
        val_sdf = validation_dataset.read.to_snowpark_dataframe()

        # Derive the feature column list dynamically so this script stays in
        # sync with the upstream feature engineering pipeline automatically.
        feature_cols = [c for c in train_sdf.columns if c not in _SPINE_COLS]
        print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]} ...")

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

        # ── 4. Train Snowflake-managed XGBoost ────────────────────────────────
        # ``snowflake.ml.modeling.xgboost.XGBRegressor`` wraps the open-source
        # XGBoost library with a Snowpark DataFrame interface.  ``input_cols``
        # and ``label_cols`` replace sklearn's positional X/y convention,
        # making column selection explicit and audit-friendly.
        # ``sample_weight_col`` applies exposure weighting — standard practice
        # for actuarial models where policies with partial-year exposure should
        # contribute proportionally to the loss function.
        # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/modeling/overview
        gbm = XGBRegressor(
            input_cols=feature_cols,
            label_cols=["PURE_PREMIUM"],
            output_cols=["PREDICTED_PURE_PREMIUM"],
            sample_weight_col="EXPOSURE",
            random_state=0,
            **{k: v for k, v in hparams.items() if k not in ("ds_version",)},
        )
        gbm.fit(train_sdf)
        print("Model training complete")

        # ── 5. Evaluate ───────────────────────────────────────────────────────
        # Evaluation metrics are computed as Snowpark SQL aggregations that run
        # entirely inside Snowflake — no validation rows are transferred to the
        # client.  This is critical for large portfolios where pulling the full
        # validation set would be slow and expensive.
        preds_sdf = gbm.predict(val_sdf)

        metrics_row = preds_sdf.select(
            F.sqrt(
                F.avg((F.col("PURE_PREMIUM") - F.col("PREDICTED_PURE_PREMIUM")) ** 2)
            ).alias("RMSE"),
            F.avg(F.abs(F.col("PURE_PREMIUM") - F.col("PREDICTED_PURE_PREMIUM"))).alias(
                "MAE"
            ),
        ).collect()[0]

        val_rmse = float(metrics_row["RMSE"])
        val_mae = float(metrics_row["MAE"])
        print(f"Validation RMSE : {val_rmse:>10.4f}")
        print(f"Validation MAE  : {val_mae:>10.4f}")

        # ``log_metrics`` stores numeric values in the run record.  Use the
        # ``step`` parameter (default 0) for time-series metrics such as
        # per-epoch training loss.
        tracker.log_metrics({"val_rmse": val_rmse, "val_mae": val_mae})

        # ── 6. Generate diagnostic plots ──────────────────────────────────────
        # Plots are saved to /tmp then attached to the model version as
        # user_files so they travel with the model in the registry.
        df_val = val_sdf.to_pandas()
        df_val["PREDICTED_PURE_PREMIUM"] = gbm.predict(val_sdf).to_pandas()[
            "PREDICTED_PURE_PREMIUM"
        ]
        plot_paths = generate_diagnostic_plots(df_val, "GBM")

        # ── 7. Log model ─────────────────────────────────────────────────────
        # ``tracker.log_model`` does two things in one call:
        #   (a) registers the fitted model as a versioned entry in the Snowflake
        #       Model Registry (visible under AI & ML → Models in Snowsight), and
        #   (b) links that model version back to this experiment run so the run
        #       record shows which model it produced.
        # ``enable_explainability=True`` pre-computes SHAP infrastructure so
        # ``mv.run(df, function_name="explain")`` works without re-training.
        # ``target_platforms`` controls where the model can be served:
        #   WAREHOUSE  → mv.run() for batch scoring on a virtual warehouse
        #   SNOWPARK_CONTAINER_SERVICES → mv.run_batch() on a compute pool
        # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview
        mv = tracker.log_model(
            model=gbm,
            model_name="ACTUARIAL_GBM",
            version_name=model_version,
            comment="XGBoost GBM for homeowners pure premium pricing.",
            metrics={"val_rmse": val_rmse, "val_mae": val_mae},
            options={"enable_explainability": True},
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            sample_input_data=training_dataset.read.to_snowpark_dataframe()
            .select(feature_cols)
            .limit(10),
            user_files={"artifacts": plot_paths},
        )  # type: ignore

        for path in plot_paths:
            tracker.log_artifact(path)

        # Generate SHAP-based explanations using the Model Registry's built-in
        # explainability function (enabled above via enable_explainability=True).
        # The returned Snowpark DataFrame contains per-feature SHAP values for
        # each validation row — useful for regulatory feature importance exhibits.
        mv.run(val_sdf, function_name="explain")
        print(f"Registered model ACTUARIAL_GBM {model_version}")


if __name__ == "__main__":
    # When running as a Snowflake ML Job, SNOWFLAKE_SERVICE_NAME is injected
    # automatically into the container environment by the SPCS runtime.  Using
    # it as the default model version means every job submission produces a
    # uniquely named model version (e.g. "TRAIN_4D0EA6A1_UGW82GXDBQB") with
    # zero manual bookkeeping.  The version is also the experiment run name,
    # so the model version and the run that produced it share the same ID.
    # Defaults to "v1" if not running as an ML Job.
    default_version = os.environ.get("SNOWFLAKE_SERVICE_NAME", "v1")

    parser = argparse.ArgumentParser(description="Train actuarial GBM model.")
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

    session = create_session()
    train(
        session,
        ds_version=args.ds_version,
        model_version=args.model_version,
        n_estimators=args.n_estimators,
    )
