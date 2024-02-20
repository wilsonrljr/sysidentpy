"""Plotting methods."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def plot_results(
    y: np.ndarray,
    *,
    yhat: np.ndarray,
    n: int = 100,
    title: str = "Free run simulation",
    xlabel: str = "Samples",
    ylabel: str = r"y, $\hat{y}$",
    data_color: str = "#1f77b4",
    model_color: str = "#ff7f0e",
    marker: str = "o",
    model_marker: str = "*",
    linewidth: float = 1.5,
    figsize: Tuple[int, int] = (10, 6),
    style: str = "default",
    facecolor: str = "white",
) -> None:
    """Plot the results of a simulation.

    Parameters
    ----------
    y : np.ndarray
        True data values.
    yhat : np.ndarray
        Model predictions.
    n : int
        Number of samples to plot.
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    data_color : str
        Color for the data line.
    model_color : str
        Color for the model line.
    marker : str
        Marker style for the data line.
    model_marker : str
        Marker style for the model line.
    linewidth : float
        Line width for both lines.
    figsize : Tuple[int, int]
        Figure size (width, height).
    style : str
        Matplotlib style.
    facecolor : str
        Figure facecolor.

    """
    if len(y) == 0 or len(yhat) == 0:
        raise ValueError("Arrays must have at least 1 samples.")

    # Set Matplotlib style and figure properties
    plt.style.use(style)
    plt.rcParams["axes.facecolor"] = facecolor

    _, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.plot(
        y[:n], c=data_color, alpha=1, marker=marker, label="Data", linewidth=linewidth
    )
    ax.plot(
        yhat[:n], c=model_color, marker=model_marker, label="Model", linewidth=linewidth
    )

    # Customize plot properties
    ax.set_title(title, fontsize=18)
    ax.legend()
    ax.tick_params(labelsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.show()


def plot_residues_correlation(
    data=None,
    *,
    figsize: Tuple[int, int] = (10, 6),
    n: int = 100,
    style: str = "default",
    facecolor: str = "white",
    title: str = "Residual Analysis",
    ylabel: str = "Correlation",
) -> None:
    """Plot the residual validation."""
    plt.style.use(style)
    plt.rcParams["axes.facecolor"] = facecolor
    _, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.plot(data[0][:n], color="#1f77b4")
    ax.axhspan(data[1], data[2], color="#ccd9ff", alpha=0.5, lw=0)
    ax.set_xlabel("Lag", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_ylim([-1, 1])
    ax.set_title(title, fontsize=18)
    plt.show()
