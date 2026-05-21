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

    from snowflake.ml.jobs import submit_file

    job = submit_file(
        "train.py",
        compute_pool="DEMO_POOL",
        stage_name="payload_stage",
        query_warehouse="COMPUTE_WH",   # required — USE WAREHOUSE is blocked inside the container
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
Snowpark Session:
    https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before any other plt import
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from snowflake.ml.dataset import load_dataset
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.modeling.xgboost import XGBRegressor  # type: ignore -> available on SPCS
from snowflake.snowpark import Session, functions as F

# Spine columns added by the Feature Store at dataset generation time.
# These are excluded when deriving the feature column list for training.
_SPINE_COLS = {"POLICY_ID", "PURE_PREMIUM", "EXPOSURE"}

# Snowflake Experiments groups runs into a named experiment so all training
# runs for this model are visible together in the Snowsight Experiments UI.
EXPERIMENT = "ACTUARIAL_GBM_TRAINING"

# NOTE: train.py runs as a standalone file inside a Snowflake ML Job container
# and cannot import from config.py.  Update these constants directly when
# changing environments.  See config.py for the local-script equivalents.
DATABASE = "COUNTRY_ML"
SCHEMA = "ACTUARIAL_PRICING"
ROLE = "ACCOUNTADMIN"
WAREHOUSE = "COMPUTE_WH"


def lorenz_curve(y_true, y_pred, exposure):
    """Compute the Lorenz curve for a pricing model's risk-discrimination ability.

    Policies are ranked from safest (lowest predicted risk) to riskiest, then
    cumulative exposure and cumulative claim amounts are computed along that
    ranking.  A perfect model would rank all high-loss policies last; a random
    model produces the diagonal (no discrimination).

    The **Gini index** = 1 − 2 × AUC(lorenz_curve), where AUC is computed via
    ``sklearn.metrics.auc``.  Higher Gini indicates better risk separation.
    Gini curves are a standard exhibit in actuarial pricing reviews and
    regulatory model-filing documentation.

    Args:
        y_true:   Array-like of observed pure premiums (or claim amounts).
        y_pred:   Array-like of model-predicted pure premiums used for ranking.
        exposure: Array-like of exposure weights (policy-years).

    Returns:
        Tuple ``(cum_exposure, cum_claims)``, both normalised to ``[0, 1]``,
        suitable for passing directly to ``matplotlib.pyplot.plot`` and
        ``sklearn.metrics.auc``.
    """
    y_true, y_pred, exposure = map(np.asarray, [y_true, y_pred, exposure])
    rank = np.argsort(y_pred)
    cum_claims = np.cumsum(y_true[rank] * exposure[rank])
    cum_claims /= cum_claims[-1]
    cum_exposure = np.cumsum(exposure[rank])
    cum_exposure /= cum_exposure[-1]
    return cum_exposure, cum_claims


def double_lift_chart(
    df_test, predictions_dict, weight="EXPOSURE", y_true="PurePremium", n_bins=10
):
    """Produce a double-lift chart — the standard actuarial model-validation exhibit.

    Each model in ``predictions_dict`` receives its own subplot.  Test policies
    are ranked into ``n_bins`` deciles by predicted pure premium (1 = safest,
    ``n_bins`` = riskiest).  Within each decile, exposure-weighted *predicted*
    and *observed* pure premiums are plotted side-by-side.

    **What to look for:**

    - Both lines should rise monotonically left → right (good risk ordering).
    - Predicted and observed lines should track closely (good calibration).
    - Large gaps in specific deciles reveal where the model misfits.

    This chart is a required exhibit in most U.S. state rate-filing packages
    and is used by internal pricing committees to validate model relativities
    before deployment.

    Args:
        df_test:          Test-set DataFrame.  Must contain ``weight`` and
                          ``y_true`` columns.
        predictions_dict: ``{label: predictions_array}`` mapping.  One subplot
                          is generated per entry.
        weight:           Column name of the exposure weight.
                          Defaults to ``"EXPOSURE"``.
        y_true:           Column name of the observed pure premium.
                          Defaults to ``"PurePremium"`` (add an alias column
                          if your DataFrame uses a different name).
        n_bins:           Number of risk deciles.  Defaults to ``10``.

    Returns:
        ``matplotlib.figure.Figure`` containing all subplots.  Call
        ``fig.savefig(path)`` to persist, then pass the path to
        ``tracker.log_artifact(path)`` to attach it to a Snowflake
        Experiments run.
    """
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, y_pred) in zip(axes, predictions_dict.items()):
        tmp = df_test[[weight, y_true]].copy()
        tmp["predicted"] = np.asarray(y_pred)
        tmp["decile"] = (
            pd.qcut(tmp["predicted"], q=n_bins, labels=False, duplicates="drop") + 1
        )
        agg = tmp.groupby("decile").apply(
            lambda g: pd.Series(
                {
                    "Predicted": np.average(g["predicted"], weights=g[weight]),
                    "Observed": np.average(g[y_true], weights=g[weight]),
                }
            )
        )
        x = agg.index.astype(int)
        ax.plot(
            x,
            agg["Predicted"],
            marker="o",
            color="steelblue",
            label="Predicted pure premium",
        )
        ax.plot(
            x,
            agg["Observed"],
            marker="s",
            color="darkorange",
            linestyle="--",
            label="Observed pure premium",
        )
        ax.set_xlabel("Risk Decile  (1 = safest → 10 = riskiest)")
        ax.set_ylabel("Exposure-Weighted Pure Premium ($)")
        ax.set_title(f"Double-Lift Chart — {label}")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend()

    plt.tight_layout()
    return fig


def _get_session() -> Session:
    """Return an active Snowpark session, using context when available.

    When running inside a Snowflake Notebook or SPCS service (including ML
    Jobs), ``get_active_session()`` returns the pre-configured session with no
    credentials required — Snowflake injects the OAuth token automatically.

    For local development the fallback reads connection details from the
    ``default`` named connection in ``~/.snowflake/connections.toml``.

    References:
        https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session
        https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect
    """
    try:
        from snowflake.snowpark.context import get_active_session

        return get_active_session()
    except Exception:
        return Session.builder.configs(
            {
                "connection_name": "default",
                "role": ROLE,
                "warehouse": WAREHOUSE,
                "database": DATABASE,
                "schema": SCHEMA,
            }
        ).create()


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
    6. **Model registration** — ``tracker.log_model`` registers the fitted
       model in the Snowflake Model Registry AND links the version to this
       experiment run in a single call.  Explainability (SHAP) is enabled
       so ``mv.run(df, function_name="explain")`` works without re-training.
    7. **Artifact upload** — diagnostic plots are saved to ``/tmp`` and
       uploaded to the experiment's artifact store via ``log_artifact``; they
       appear in the Artifacts tab in Snowsight.

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
        Do NOT call ``session.use_warehouse()`` inside this function.  When
        running as an ML Job the warehouse is configured at submission time
        via the ``query_warehouse`` argument to ``submit_file``; issuing
        ``USE WAREHOUSE`` from inside the container is rejected by Snowflake.
    """
    # Set role, database, and schema explicitly so the function is portable
    # across execution environments (local, ML Job, Snowflake Notebook).
    session.sql(f"USE ROLE {ROLE}").collect()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)

    # ── 1. Experiment Tracking setup ─────────────────────────────────────────
    # ExperimentTracking is an MLflow-compatible singleton.  ``set_experiment``
    # creates the experiment object in Snowflake if it does not already exist.
    # ``start_run`` opens a run context; ``end_run`` (called in ``finally``)
    # closes it and persists all logged data.
    # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiment-tracking
    tracker = ExperimentTracking(
        session=session,
        database_name=DATABASE,
        schema_name=SCHEMA,
    )
    tracker.set_experiment(EXPERIMENT)
    tracker.start_run(run_name=model_version)

    try:
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

            # ── 7. Generate and upload diagnostic plots ───────────────────────────
        # Plots are saved to /tmp (available in both local and SPCS container
        # environments) then uploaded via ``log_artifact``.  Artifacts are
        # stored in the experiment's internal Snowflake stage and appear in the
        # Artifacts tab of the run in Snowsight.
        df_val = val_sdf.to_pandas()
        df_val["PREDICTED_PURE_PREMIUM"] = gbm.predict(val_sdf).to_pandas()[
            "PREDICTED_PURE_PREMIUM"
        ]
        # double_lift_chart expects y_true="PurePremium" (camelCase) by default.
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
        
        # ── 6. Log model ──────────────────────────────────────────────────────
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
            sample_input_data=train_sdf.select(feature_cols).limit(10),
            user_files={
                "artifacts": [
                    "/tmp/double_lift.png",
                    "/tmp/gini_lorenz.png",
                ]
            },
        )  # type: ignore



        tracker.log_artifact("/tmp/double_lift.png")
        tracker.log_artifact("/tmp/gini_lorenz.png")
        
        # Generate SHAP-based explanations using the Model Registry's built-in
        # explainability function (enabled above via enable_explainability=True).
        # The returned Snowpark DataFrame contains per-feature SHAP values for
        # each validation row — useful for regulatory feature importance exhibits.
        explanations = mv.run(val_sdf, function_name="explain")
        print(f"Registered model ACTUARIAL_GBM {model_version}")

    finally:
        # end_run() must be called even if training fails so the run is not
        # left open in the Experiments UI.
        tracker.end_run()


if __name__ == "__main__":
    # When running as a Snowflake ML Job, SNOWFLAKE_SERVICE_NAME is injected
    # automatically into the container environment by the SPCS runtime.  Using
    # it as the default model version means every job submission produces a
    # uniquely named model version (e.g. "TRAIN_4D0EA6A1_UGW82GXDBQB") with
    # zero manual bookkeeping.  The version is also the experiment run name,
    # so the model version and the run that produced it share the same ID.
    # Falls back to "v1" for local runs where the env var is not set.
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

    session = _get_session()
    train(
        session,
        ds_version=args.ds_version,
        model_version=args.model_version,
        n_estimators=args.n_estimators,
    )
