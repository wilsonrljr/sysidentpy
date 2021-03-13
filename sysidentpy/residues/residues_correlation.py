# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt


class ResiduesAnalysis:
    """Residues analysis for Polynomial NARX model."""

    def residuals(self, X, y, yhat):
        """Performs the residual analysis of output to validate model.

        Parameters
        ----------
        y : array-like of shape = n_samples
            The target data used in the identification process.
        yhat : array-like of shape = n_samples
            The prediction values of the identification process.
        X : ndarray of floats
            The input data.

        Returns
        -------
        output_autocorr : ndarray of floats:
            1st column - Residuals normalized autocorrelation.
            2nd/3rd columns - Superior and inferior limits of a
            95% confidence interval.
        output_crosscorr : ndarray of floats:
            1st column - Correlation between residuals and input.
            2nd/3rd columns - Superior and inferior limits of a
            95% confidence interval.

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> autocorr(y)
        [62.25 11.5   2.5  21.  ]

        """
        e, e_acf, unnormalized_e_acf = self._residuals_acf(y, yhat)

        xe_ccf = self._input_ccf(X[:, 0], e, len(unnormalized_e_acf))
        ex = e * X[:, 0]
        ee = e ** 2
        x2line = (X[:, 0] ** 2) - (X[:, 0] ** 2).mean()
        eeline = (e ** 2) - (e ** 2).mean()
        e_ex = self._input_ccf(e, ex, len(unnormalized_e_acf))
        x2line_e = self._input_ccf(x2line, e, len(unnormalized_e_acf))
        x2line_ee = self._input_ccf(x2line, ee, len(unnormalized_e_acf))
        ye = y * e.reshape(-1, 1)
        yeline = (y * e.reshape(-1, 1)) - (y * e.reshape(-1, 1)).mean()

        yeline_x2line = self._input_ccf(yeline, x2line, len(unnormalized_e_acf))
        lam = np.sqrt(sum((ee - ee.mean()) ** 2) / sum((ye - ye.mean()) ** 2))
        yeline_eeline = self._input_ccf(yeline, eeline, len(unnormalized_e_acf))
        return (
            e_acf,
            xe_ccf,
            [yeline_x2line, yeline_eeline, e_ex, x2line_e, x2line_ee],
            lam,
        )

    def _input_ccf(self, X, e, len_confidence):
        xe_ccf = np.zeros((int(np.floor(len(X) / 2)), 3), dtype=float)

        xe_ccf[:, 0] = self._normalized_correlation(X, e)

        xe_ccf[:, 1] = np.ones(len(xe_ccf[:, 0])) * (1.96 / np.sqrt(len_confidence))

        xe_ccf[:, 2] = xe_ccf[:, 1] * (-1)
        return xe_ccf

    def _residuals_acf(self, y, yhat):
        e = (y - yhat).flatten()
        unnormalized_e_acf = np.correlate(e, e, mode="full")
        half_of_simmetry_autocorr = int(np.floor(unnormalized_e_acf.size / 2))

        e_acf = np.zeros(
            (len(unnormalized_e_acf) - half_of_simmetry_autocorr, 3), dtype=float
        )

        e_acf[:, 0] = (
            unnormalized_e_acf[half_of_simmetry_autocorr:]
            / unnormalized_e_acf[half_of_simmetry_autocorr]
        )

        e_acf[:, 1] = np.ones(len(e_acf[:, 0])) * (
            1.96 / np.sqrt(len(unnormalized_e_acf))
        )

        e_acf[:, 2] = e_acf[:, 1] * (-1)
        return e, e_acf, unnormalized_e_acf

    def _normalized_correlation(self, signal1, signal2):
        """Compute the normalized correlation between two signals.

        Parameters
        ----------
        signal1 : array-like of shape = n_samples.
        signal2 : array-like of shape = n_samples.

        Returns
        -------
        ruy : ndarray of floats:
            The normalized cross correlation between the two signals.

        """
        y = (signal1 - np.mean(signal1)).flatten()
        u = (signal2 - np.mean(signal2)).flatten()
        t = int(np.floor(len(signal1) / 2))
        ruy = np.array(np.zeros(t))
        ruy[0] = np.sum(y * u) / (np.sqrt(np.sum(y ** 2)) * np.sqrt(np.sum(u ** 2)))

        for i in range(1, t):
            y = (signal1 - np.mean(signal1[:i])).flatten()
            u = (signal2 - np.mean(signal2[i:])).flatten()
            ruy[i] = np.sum(y[:-i] * u[i:]) / (
                np.sqrt(np.sum(y[:-i] ** 2)) * np.sqrt(np.sum(u[i:] ** 2))
            )

        return ruy

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
            y[self.max_lag : n],
            c="#1f77b4",
            alpha=1,
            marker="o",
            label="Data",
            linewidth=1.5,
        )
        ax.plot(
            yhat[self.max_lag : n],
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
