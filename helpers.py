from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_tweedie_deviance


def plot_obs_pred(df, feature, weight, observed, predicted,
                  y_label=None, title=None, ax=None, fill_legend=False):
    """Observed vs. predicted, aggregated by a rating factor.
    Exposure distribution is shown on a secondary y-axis so the two
    quantities are not conflated on the same scale."""
    df_ = df[[feature, weight]].copy()
    df_["observed"]  = df[observed]  * df[weight]
    df_["predicted"] = predicted     * df[weight]
    df_ = (
        df_.groupby(feature)[[weight, "observed", "predicted"]]
        .sum()
        .assign(observed  = lambda x: x["observed"]  / x[weight])
        .assign(predicted = lambda x: x["predicted"] / x[weight])
    )
    ax = df_[["observed", "predicted"]].plot(style=".", ax=ax)
    ax.set(ylabel=y_label, title=title or "Observed vs. Predicted")

    ax2 = ax.twinx()
    ax2.fill_between(
        df_.index, 0, df_[weight],
        color="green", alpha=0.08, label=f"{feature} exposure",
    )
    ax2.set_ylabel("Exposure (policy-years)", color="green", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="green", labelsize=7)
    ax2.set_ylim(bottom=0)

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    if fill_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left")

    return ax


def score_estimator(estimator, X_train, X_test, df_train, df_test,
                    target, weights, tweedie_powers=None):
    """Evaluate an estimator on train/test with common actuarial metrics."""
    metrics = [
        ("D² explained",       None),
        ("mean abs. error",    mean_absolute_error),
        ("mean squared error", mean_squared_error),
    ]
    if tweedie_powers:
        metrics += [
            (f"mean Tweedie dev p={p:.4f}", partial(mean_tweedie_deviance, power=p))
            for p in tweedie_powers
        ]
    res = []
    for label, X, df in [("train", X_train, df_train), ("test", X_test, df_test)]:
        y, w = df[target], df[weights]
        for score_label, metric in metrics:
            if isinstance(estimator, tuple) and len(estimator) == 2:
                y_pred = estimator[0].predict(X) * estimator[1].predict(X)
            else:
                y_pred = estimator.predict(X)
            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                score = estimator.score(X, y, sample_weight=w)
            else:
                score = metric(y, y_pred, sample_weight=w)
            res.append({"subset": label, "metric": score_label, "score": score})
    return (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        [["train", "test"]]
    )


def lorenz_curve(y_true, y_pred, exposure):
    """Lorenz curve: cumulative exposure vs. cumulative claims ranked by model."""
    y_true, y_pred, exposure = map(np.asarray, [y_true, y_pred, exposure])
    rank         = np.argsort(y_pred)
    cum_claims   = np.cumsum(y_true[rank] * exposure[rank])
    cum_claims  /= cum_claims[-1]
    cum_exposure = np.cumsum(exposure[rank])
    cum_exposure /= cum_exposure[-1]
    return cum_exposure, cum_claims


def double_lift_chart(df_test, predictions_dict,
                      weight="EXPOSURE", y_true="PurePremium", n_bins=10):
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
        tmp["decile"]    = (
            pd.qcut(tmp["predicted"], q=n_bins, labels=False, duplicates="drop") + 1
        )
        agg = tmp.groupby("decile").apply(
            lambda g: pd.Series({
                "Predicted": np.average(g["predicted"], weights=g[weight]),
                "Observed":  np.average(g[y_true],     weights=g[weight]),
            })
        )
        x = agg.index.astype(int)
        ax.plot(x, agg["Predicted"], marker="o", color="steelblue",
                label="Predicted pure premium")
        ax.plot(x, agg["Observed"],  marker="s", color="darkorange",
                linestyle="--", label="Observed pure premium")
        ax.set_xlabel("Risk Decile  (1 = safest → 10 = riskiest)")
        ax.set_ylabel("Exposure-Weighted Pure Premium ($)")
        ax.set_title(f"Double-Lift Chart — {label}")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend()

    plt.tight_layout()
    return fig
