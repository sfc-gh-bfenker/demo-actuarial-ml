"""
Actuarial diagnostic plotting and scoring utilities.

Adapted from the scikit-learn French motor insurance tutorial:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html

These helpers operate on plain pandas DataFrames and are designed to work
directly with data pulled from Snowflake Feature Store datasets via
``dataset.read.to_pandas()``.  They are presentation-layer utilities only —
no Snowflake dependencies are introduced here.

Key outputs used in regulatory filings:
- ``double_lift_chart``: primary actuarial model-validation exhibit
- ``lorenz_curve`` / Gini index: risk-discrimination summary for pricing reviews
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


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
        ``fig.savefig(path)`` to persist; pass to
        ``tracker.log_artifact(path)`` to attach to a Snowflake experiment run.
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
