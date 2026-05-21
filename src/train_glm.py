"""
Actuarial GLM training pipeline for homeowners pure premium.

Reads pre-built ACTUARIAL_TRAINING and ACTUARIAL_VALIDATION datasets from the
Snowflake Feature Store, trains a Snowflake-managed Tweedie GLM, tracks the
run in Snowflake Experiments, and registers the model in the Model Registry.

Model selection rationale
-------------------------
Pure premium (expected loss per unit of exposure) is the product of frequency
and severity.  Its distribution is compound Poisson-Gamma, which belongs to the
**Tweedie family** with variance power *p* ∈ (1, 2):

* *p* = 1  → Poisson  (pure frequency, all claims equal size)
* *p* = 2  → Gamma    (pure severity, no zero claims)
* *p* = 1.5 → Compound Poisson-Gamma (standard actuarial pure premium)

Using a **log link** produces a multiplicative pricing plan — the industry
standard — where each rating factor is applied as a percentage multiplier to
the base rate.  Regulatory submissions expect this structure.

The Tweedie GLM serves as the actuarial benchmark ("classical" model) that the
GBM in ``train.py`` is compared against.  Side-by-side evaluation on the
double-lift chart and Lorenz curve quantifies the lift from moving to a
gradient-boosted model.

References
----------
Tweedie GLM in actuarial pricing:
    Smyth & Jørgensen (2002), "Fitting Tweedie's compound Poisson model"
    https://link.springer.com/article/10.1023/A:1020446010384
scikit-learn TweedieRegressor:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html

Usage
-----
Local::

    python train_glm.py
    python train_glm.py --ds-version 2 --model-version v2

As a Snowflake ML Job::

    from snowflake.ml.jobs import submit_file

    job = submit_file(
        "train_glm.py",
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
Snowpark ML Modeling (TweedieRegressor):
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
from sklearn.linear_model import TweedieRegressor as _SkTweedie
from sklearn.metrics import auc, mean_squared_error
from sklearn.model_selection import KFold
from snowflake.ml.dataset import load_dataset
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.modeling.linear_model import TweedieRegressor  # type: ignore -> available on SPCS
from snowflake.snowpark import Session, functions as F

# Spine columns added by the Feature Store at dataset generation time.
# These are excluded when deriving the feature column list for training.
_SPINE_COLS = {"POLICY_ID", "PURE_PREMIUM", "EXPOSURE"}

# Snowflake Experiments groups runs into a named experiment so all training
# runs for this model are visible together in the Snowsight Experiments UI.
EXPERIMENT = "ACTUARIAL_GLM_TRAINING"

# NOTE: train_glm.py runs as a standalone file inside a Snowflake ML Job
# container and cannot import from config.py.  Update these constants directly
# when changing environments.  See config.py for the local-script equivalents.
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
    alpha: float = 0.1,
    n_folds: int = 5,
) -> None:
    """Run the end-to-end actuarial Tweedie GLM training pipeline.

    Steps performed:

    1. **Experiment Tracking setup** — creates or resumes a named experiment
       and opens a run.  All params, metrics, artifacts, and the model version
       are linked to this run in the Snowsight Experiments UI.
    2. **Feature Store dataset load** — reads the pre-built, versioned
       ACTUARIAL_TRAINING / ACTUARIAL_VALIDATION datasets directly from
       Snowflake without any data egress to the local machine.
    3. **Hyperparameter logging** — stores all training parameters as
       key-value pairs visible in the run comparison view.
    4. **Snowflake-managed Tweedie GLM training** — ``TweedieRegressor`` from
       ``snowflake.ml.modeling.linear_model`` wraps scikit-learn's Tweedie
       estimator with a Snowpark DataFrame interface; training executes on
       Snowflake compute.  ``power=1.5`` sets the compound Poisson-Gamma
       distribution; ``link='log'`` produces a multiplicative pricing plan.
    4b. **K-fold cross-validation** — the training data is pulled to pandas
        once and split into ``n_folds`` folds using scikit-learn ``KFold``.
        Per-fold RMSE (exposure-weighted) is computed with the native sklearn
        estimator to avoid redundant Snowflake round-trips.  Mean and std of
        CV RMSE are logged as experiment metrics for hyperparameter comparison.
    5. **Server-side evaluation** — RMSE and MAE are computed as Snowpark SQL
       aggregations; no rows are transferred to the client for scoring.
    6. **Model registration** — ``tracker.log_model`` registers the fitted
       model in the Snowflake Model Registry AND links the version to this
       experiment run in a single call.
    7. **Artifact upload** — diagnostic plots (double-lift, Lorenz/Gini) are
       saved to ``/tmp`` and uploaded via ``log_artifact``; they appear in the
       Artifacts tab of the run in Snowsight.

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
        alpha:         L2 (ridge) regularisation strength for the Tweedie GLM.
                       Larger values shrink coefficients toward zero, which
                       reduces overfitting on sparse indicator features.
                       Corresponds directly to scikit-learn's ``alpha``
                       parameter on ``TweedieRegressor``.
        n_folds:       Number of folds for k-fold cross-validation on the
                       training set.  Set to 0 or 1 to skip CV.

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
            # Tweedie variance power: 1.5 → compound Poisson-Gamma, the actuarial
            # standard for pure premium.  p=1 is pure Poisson (frequency only);
            # p=2 is pure Gamma (severity only, no zeros).
            power=1.5,
            # Log link produces a multiplicative rating plan — predicted pure
            # premium = exp(β₀ + β₁x₁ + … + βₙxₙ) — matching the structure
            # used in filed rate manuals.
            link="log",
            # L2 regularisation: shrinks coefficients of correlated indicator
            # variables (OHE region/territory flags) toward a common mean,
            # reducing variance on thin rating cells.
            alpha=alpha,
            max_iter=1000,
            n_folds=n_folds,
            ds_version=ds_version,
        )
        tracker.log_params(hparams)

        # ── 3b. K-fold cross-validation ───────────────────────────────────────
        # Cross-validation is run on the training set using the native sklearn
        # estimator to avoid creating N Snowflake model objects.  The training
        # data is pulled to pandas once; all fold splits and fits happen
        # locally.  This is appropriate for a GLM whose fit is fast and whose
        # coefficients are the primary diagnostic output.
        #
        # CV RMSE mean/std are logged as experiment metrics so runs with
        # different ``alpha`` values can be compared directly in Snowsight.
        if n_folds > 1:
            print(f"Running {n_folds}-fold cross-validation ...")
            train_pd = train_sdf.to_pandas()
            X_cv = train_pd[feature_cols].values
            y_cv = train_pd["PURE_PREMIUM"].values
            w_cv = train_pd["EXPOSURE"].values

            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_rmses = []
            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_cv), start=1):
                fold_model = _SkTweedie(
                    power=hparams["power"],
                    alpha=alpha,
                    link=hparams["link"],
                    max_iter=hparams["max_iter"],
                )
                fold_model.fit(X_cv[tr_idx], y_cv[tr_idx], sample_weight=w_cv[tr_idx])
                fold_preds = fold_model.predict(X_cv[val_idx])
                fold_rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            y_cv[val_idx], fold_preds, sample_weight=w_cv[val_idx]
                        )
                    )
                )
                fold_rmses.append(fold_rmse)
                print(f"  Fold {fold_idx}/{n_folds}  RMSE={fold_rmse:.4f}")

            cv_rmse_mean = float(np.mean(fold_rmses))
            cv_rmse_std = float(np.std(fold_rmses))
            print(f"CV RMSE  {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
            tracker.log_metrics({"cv_rmse_mean": cv_rmse_mean, "cv_rmse_std": cv_rmse_std})

        # ── 4. Train Snowflake-managed Tweedie GLM ────────────────────────────
        # ``snowflake.ml.modeling.linear_model.TweedieRegressor`` wraps
        # scikit-learn's Tweedie estimator with a Snowpark DataFrame interface.
        # ``input_cols`` and ``label_cols`` replace sklearn's positional X/y
        # convention, making column selection explicit and audit-friendly.
        # ``sample_weight_col`` applies exposure weighting — standard practice
        # for actuarial models where policies with partial-year exposure should
        # contribute proportionally to the log-likelihood.
        # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/modeling/overview
        glm = TweedieRegressor(
            input_cols=feature_cols,
            label_cols=["PURE_PREMIUM"],
            output_cols=["PREDICTED_PURE_PREMIUM"],
            sample_weight_col="EXPOSURE",
            **{k: v for k, v in hparams.items() if k not in ("ds_version", "n_folds")},
        )
        glm.fit(train_sdf)
        print("Model training complete")

        # ── 5. Evaluate ───────────────────────────────────────────────────────
        # Evaluation metrics are computed as Snowpark SQL aggregations that run
        # entirely inside Snowflake — no validation rows are transferred to the
        # client.  This is critical for large portfolios where pulling the full
        # validation set would be slow and expensive.
        preds_sdf = glm.predict(val_sdf)

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

        # ── 6. Log model ──────────────────────────────────────────────────────
        # ``tracker.log_model`` does two things in one call:
        #   (a) registers the fitted model as a versioned entry in the Snowflake
        #       Model Registry (visible under AI & ML → Models in Snowsight), and
        #   (b) links that model version back to this experiment run so the run
        #       record shows which model it produced.
        # ``target_platforms`` controls where the model can be served:
        #   WAREHOUSE  → mv.run() for batch scoring on a virtual warehouse
        #   SNOWPARK_CONTAINER_SERVICES → mv.run_batch() on a compute pool
        # Note: linear models use SHAP's LinearExplainer rather than TreeSHAP.
        # enable_explainability is omitted here as support varies by model type;
        # coefficient inspection (glm.to_sklearn().coef_) provides interpretability
        # for GLMs without requiring SHAP.
        # Docs: https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview
        mv = tracker.log_model(
            model=glm,
            model_name="ACTUARIAL_GLM",
            version_name=model_version,
            comment="Tweedie GLM (power=1.5, log link) for homeowners pure premium pricing.",
            metrics={"val_rmse": val_rmse, "val_mae": val_mae},
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        )  # type: ignore

        # ── 7. Generate and upload diagnostic plots ───────────────────────────
        # Plots are saved to /tmp (available in both local and SPCS container
        # environments) then uploaded via ``log_artifact``.  Artifacts are
        # stored in the experiment's internal Snowflake stage and appear in the
        # Artifacts tab of the run in Snowsight.
        df_val = val_sdf.to_pandas()
        df_val["PREDICTED_PURE_PREMIUM"] = glm.predict(val_sdf).to_pandas()[
            "PREDICTED_PURE_PREMIUM"
        ]
        # double_lift_chart expects y_true="PurePremium" (camelCase) by default.
        df_val["PurePremium"] = df_val["PURE_PREMIUM"]

        # double_lift_chart creates and returns its own figure — no plt.subplots()
        # call is needed before it.
        fig_lift = double_lift_chart(
            df_val, {"GLM": df_val["PREDICTED_PURE_PREMIUM"].values}, n_bins=10
        )
        fig_lift.savefig("/tmp/double_lift.png", dpi=150, bbox_inches="tight")
        plt.close(fig_lift)

        fig_gini, ax = plt.subplots(figsize=(8, 8))
        cum_exp, cum_claims = lorenz_curve(
            df_val["PURE_PREMIUM"], df_val["PREDICTED_PURE_PREMIUM"], df_val["EXPOSURE"]
        )
        gini = 1 - 2 * auc(cum_exp, cum_claims)
        ax.plot(cum_exp, cum_claims, label=f"GLM (Gini={gini:.3f})")
        ax.plot([0, 1], [0, 1], "--k", label="Random")
        ax.legend()
        fig_gini.savefig("/tmp/gini_lorenz.png", dpi=150, bbox_inches="tight")
        plt.close(fig_gini)

        tracker.log_artifact("/tmp/double_lift.png")
        tracker.log_artifact("/tmp/gini_lorenz.png")

        print(f"Registered model ACTUARIAL_GLM {model_version}")

    finally:
        # end_run() must be called even if training fails so the run is not
        # left open in the Experiments UI.
        tracker.end_run()


if __name__ == "__main__":
    # When running as a Snowflake ML Job, SNOWFLAKE_SERVICE_NAME is injected
    # automatically into the container environment by the SPCS runtime.  Using
    # it as the default model version means every job submission produces a
    # uniquely named model version (e.g. "TRAIN_GLM_4D0EA6A1_UGW82GXDBQB") with
    # zero manual bookkeeping.  The version is also the experiment run name,
    # so the model version and the run that produced it share the same ID.
    # Falls back to "v1" for local runs where the env var is not set.
    default_version = os.environ.get("SNOWFLAKE_SERVICE_NAME", "v1")

    parser = argparse.ArgumentParser(description="Train actuarial Tweedie GLM model.")
    parser.add_argument(
        "--ds-version", default="1", help="Feature Store dataset version"
    )
    parser.add_argument(
        "--model-version",
        default=default_version,
        help="Experiment run name / Model Registry version",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="L2 regularisation strength for the Tweedie GLM (default: 0.1).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        dest="n_folds",
        help="Number of k-fold CV splits on the training set (default: 5). Set to 1 to skip.",
    )
    args = parser.parse_args()

    session = _get_session()
    train(
        session,
        ds_version=args.ds_version,
        model_version=args.model_version,
        alpha=args.alpha,
        n_folds=args.n_folds,
    )
