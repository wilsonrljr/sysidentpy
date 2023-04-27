# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def plot_results(
    y=None,
    *,
    yhat=None,
    figsize=(10, 6),
    n=100,
    style="seaborn-white",
    facecolor="white",
    title="Free run simulation",
):
    plt.style.use(style)
    plt.rcParams["axes.facecolor"] = facecolor
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.plot(y[:n], c="#1f77b4", alpha=1, marker="o", label="Data", linewidth=1.5)
    ax.plot(yhat[:n], c="#ff7f0e", marker="*", label="Model", linewidth=1.5)

    ax.set_title(title, fontsize=18)
    ax.legend()
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Samples", fontsize=14)
    ax.set_ylabel("y, $\hat{y}$", fontsize=14)
    plt.show()


def plot_residues_correlation(
    data=None,
    *,
    figsize=(10, 6),
    n=100,
    style="seaborn-white",
    facecolor="white",
    title="Residual Analysis",
    ylabel="Correlation",
):
    plt.style.use(style)
    plt.rcParams["axes.facecolor"] = facecolor
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.plot(data[0], color="#1f77b4")
    ax.axhspan(data[1], data[2], color="#ccd9ff", alpha=0.5, lw=0)
    ax.set_xlabel("Lag", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_ylim([-1, 1])
    ax.set_title(title, fontsize=18)
    plt.show()
