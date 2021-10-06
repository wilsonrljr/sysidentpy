# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

def plot_result(self, y, yhat, e_acf, xe_ccf, figsize=(10, 8), n=100):
    """Plot the free run simulation and residues analysis.

    Parameters
    ----------
    y : array-like of shape = n_samples
        The target data used in the identification process.
    yhat : array-like of shape = n_samples
        The prediction values of the identification process.
    e_acf : ndarray of floats:
        1st column - Residuals normalized autocorrelation.
        2nd/3rd columns - Superior and inferior limits of a
        95% confidence interval.
    xe_ccf : ndarray of floats:
        1st column - Correlation between residuals and input.
        2nd/3rd columns - Superior and inferior limits of a
        95% confidence interval.

    """
    plt.style.use("seaborn-white")
    plt.rcParams["axes.facecolor"] = "white"

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, facecolor="white")
    fig.subplots_adjust(hspace=0.7)
    for ax, feature in zip(axes.flatten()[2:], [e_acf, xe_ccf]):
        ax.plot(feature[:, 0], color="#1f77b4")
        ax.axhspan(feature[0, 1], feature[0, 2], color="#ccd9ff", alpha=0.5, lw=0)
        ax.set_xlabel("Lag", fontsize=12)
        ax.set_ylabel("Cross Correlation: ee, ex", fontsize=12)
        # ax = plt.gca()
        ax.set_ylim([-1, 1])
        # ax.grid(color="grey", linestyle="-.", alpha=0.1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        fig.tight_layout()

    ax = plt.subplot(211)
    ax.plot(
        y[:n],
        c="#1f77b4",
        alpha=1,
        marker="o",
        label="Data",
        linewidth=1.5,
    )
    ax.plot(
        yhat[:n],
        c="#ff7f0e",
        marker="*",
        # linestyle='dashed',
        label="Model",
        linewidth=1.5,
    )
    ax.set_title("Free run simulation", fontsize=18)
    ax.legend()
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Samples", fontsize=14)
    ax.set_ylabel("y, yhat", fontsize=14)
    # ax.grid(color="grey", linestyle="-.", alpha=0.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    fig.tight_layout()
    plt.show()