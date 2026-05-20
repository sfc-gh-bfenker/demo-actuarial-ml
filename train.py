"""
Actuarial GBM training pipeline for homeowners pure premium.

Reads pre-built ACTUARIAL_TRAINING and ACTUARIAL_VALIDATION datasets from the
Snowflake Feature Store, trains a Snowflake-managed XGBoost model, tracks the
run in Snowflake Experiments, and registers the model.

Usage:
    python train.py
    python train.py --ds-version 2 --model-version v2

Submit as an ML Job (warehouse must be passed via query_warehouse, not set
inside the script — USE WAREHOUSE is rejected inside the job container):

    job = submit_file(
        "train.py",
        compute_pool="DEMO_POOL",
        stage_name="payload_stage",
        query_warehouse="COMPUTE_WH",
        args=["--ds-version=1"],
        session=session,
    )
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

_SPINE_COLS = {"POLICY_ID", "PURE_PREMIUM", "EXPOSURE"}
EXPERIMENT = "ACTUARIAL_GBM_TRAINING"

DATABASE = "COUNTRY_BANK_DEMO_DB"
SCHEMA = "ACTUARIAL_PRICING"
ROLE = "COUNTRY_BANK_DEMO_ROLE"


def lorenz_curve(y_true, y_pred, exposure):
    """Lorenz curve: cumulative exposure vs. cumulative claims ranked by model."""
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
    """
    Double-lift chart: rank test policies by predicted pure premium into deciles,
    then compare exposure-weighted predicted vs. observed per decile.

    A well-calibrated model should show:
      - Monotone increase in both lines (predicted and observed)
      - Predicted and observed lines tracking closely together
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
    try:
        from snowflake.snowpark.context import get_active_session

        return get_active_session()
    except Exception:
        return Session.builder.configs(
            {
                "connection_name": "default",
                "role": ROLE,
                "warehouse": "COMPUTE_WH",
                "database": DATABASE,
                "schema": SCHEMA,
            }
        ).create()


def train(
    session: Session,
    ds_version: str = "1",
    model_version: str = "v1",
) -> None:
    """
    Read existing Feature Store datasets, train Snowflake-managed XGBoost,
    track the run with Snowflake Experiments, and register the model.

    Args:
        session:       Active Snowpark session.
        ds_version:    Version of ACTUARIAL_TRAINING / ACTUARIAL_VALIDATION to use.
        model_version: Experiment run name and Model Registry version label.
    """

    # Set role, database, schema. Do NOT call use_warehouse — when running as an
    # ML Job the warehouse must be configured via query_warehouse in submit_file;
    # issuing USE WAREHOUSE from inside the container is rejected by Snowflake.
    session.sql(f"USE ROLE {ROLE}").collect()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)

    # ── 1. Experiment tracking setup ─────────────────────────────────────────
    tracker = ExperimentTracking(
        session=session,
        database_name=DATABASE,
        schema_name=SCHEMA,
    )
    tracker.set_experiment(EXPERIMENT)
    tracker.start_run(run_name=model_version)

    try:
        # ── 2. Load pre-built Feature Store datasets ──────────────────────────
        training_dataset = load_dataset(
            session, name="ACTUARIAL_TRAINING", version=ds_version
        )
        validation_dataset = load_dataset(
            session, name="ACTUARIAL_VALIDATION", version=ds_version
        )

        train_sdf = training_dataset.read.to_snowpark_dataframe()
        val_sdf = validation_dataset.read.to_snowpark_dataframe()

        feature_cols = [c for c in train_sdf.columns if c not in _SPINE_COLS]
        print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]} ...")

        # ── 3. Log hyperparameters ────────────────────────────────────────────
        hparams = dict(
            n_estimators=200,
            learning_rate=0.05,
            max_leaves=31,
            objective="reg:squarederror",
            ds_version=ds_version,
        )
        tracker.log_params(hparams)

        # ── 4. Train Snowflake-managed XGBoost ────────────────────────────────
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

        tracker.log_metrics({"val_rmse": val_rmse, "val_mae": val_mae})

        # ── 6. Log model (registers in Model Registry and links to run) ───────
        mv = tracker.log_model(
            model=gbm,
            model_name="ACTUARIAL_GBM",
            version_name=model_version,
            comment="XGBoost GBM for homeowners pure premium pricing.",
            metrics={"val_rmse": val_rmse, "val_mae": val_mae},
            options={"enable_explainability": True},
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        )  # type: ignore

        df_val = val_sdf.to_pandas()
        df_val["PREDICTED_PURE_PREMIUM"] = gbm.predict(val_sdf).to_pandas()[
            "PREDICTED_PURE_PREMIUM"
        ]
        # double_lift_chart expects y_true="PurePremium" by default
        df_val["PurePremium"] = df_val["PURE_PREMIUM"]

        # double_lift_chart creates and returns its own figure
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

        explanations = mv.run(val_sdf, function_name="explain")
        print(f"Registered model ACTUARIAL_GBM {model_version}")

    finally:
        tracker.end_run()


if __name__ == "__main__":
    # Default model_version to the job's service name so each ML Job submission
    # produces a uniquely named version without manual incrementing.
    # SNOWFLAKE_SERVICE_NAME is injected by Snowflake into every SPCS container.
    # Falls back to "v1" for local runs.
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
    args = parser.parse_args()

    session = _get_session()
    train(session, ds_version=args.ds_version, model_version=args.model_version)
