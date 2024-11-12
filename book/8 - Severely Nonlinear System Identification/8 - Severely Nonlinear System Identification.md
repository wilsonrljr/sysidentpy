We have categorized systems into two different classes for now: **linear systems** and **nonlinear** systems. As mentioned, **linear systems** has been extensively studied with several different well-established methods available, while **nonlinear** systems is a very active field with several problems that are still open for research. Besides linear and nonlinear systems, there are the ones called **Severely Nonlinear Systems**. Severely Nonlinear Systems are the ones that exhibit highly complex and exotic dynamic behaviors like sub-harmonics, chaotic behavior and hysteresis. For now, we will focus on system with hysteresis.

## Modeling Hysteresis With Polynomial NARX Model

Hysteresis nonlinearity is a severely nonlinear behavior commonly found in electromagnetic devices, sensors, semiconductors, intelligent materials, and many more, which have memory effects between quasi-static input and output ([Visintin, A.]([Differential Models of Hysteresis](https://link.springer.com/book/10.1007/978-3-662-11557-2)), [Ahmad, I.]([Two Degree-of-Freedom Robust Digital Controller Design With Bouc-Wen Hysteresis Compensator for Piezoelectric Positioning Stage](https://ieeexplore.ieee.org/document/8316821))). A hysteretic system is one that exhibits a path-dependent behavior, meaning its response depends not only on its current state but also on its history.  In a hysteretic system, when you apply an input, the system's response (like displacement or stress) doesn't follow the same path to the starting point when you remove the input. Instead, it forms a loop-like pattern called a hysteresis loop. This is because the system have the *ability* to preserve a deformation caused by an input, characterizing a memory effect.

The identification of hysteretic systems using polynomial NARX models is typically an intriguing task because the traditional Model Structure Selection algorithms do not work properly ([Martins, S. A. M. and Aguirre, L. A.]([Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968)), [Leva, A. and Piroddi, L.]([NARX-based technique for the modelling of magneto-rheological damping devices](https://iopscience.iop.org/article/10.1088/0964-1726/11/1/309))). [Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968) presented the sufficient conditions to describe hysteresis using polynomial models by providing the concept of bounding structure $\mathcal{H}$. Polynomial NARX models with a single equilibrium can be used in a full characterization of the hysteresis behavior adopting the bounding structure concept.

The following are some of the essential concepts and formal definitions for understanding how NARX model can be used to describe systems with hysteresis.

### Continuous-time loading-unloading quasi-static signal

One important characteristic to model hysteretic systems is the input signal. A loading-unloading quasi-static signal is a periodic continuous time signal $x_t$ with period $T = (t_f - t_i)$ and frequency $\omega = 2\pi f$ where $x_t$ increases monotonically from $x_{min}$ to $x_{max}$, considering $t_i \leq t \leq t_m$ (loading) and decreases monotonically from $x_{max}$ to $x_{min}$, considering $t_m \leq t \leq t_f$ (unloading). If the loading-unloading signal changes with $\omega \rightarrow 0$, the signal is also called a quasi-static signal. Visually, this is much more simple to understand. The following image shows a continuous-time loading-unloading quasi-static signal.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/load_unloading_signal.png?raw=true)
> Figure 1. Continuous-time loading-unloading quasi-static signal, demonstrating the periodic increase and decrease of the input signal.


In this respect, [Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968) also presented the idea of transforming the inputs of the system using multi-valued functions.

> Multi-valued functions - Let $\phi (\Delta x_{k}): \mathbb{R} \rightarrow \mathbb{R}$. If~$\Delta x_{k}=x_k-x_{k-1}$, $\phi (\Delta x_{k})$ is a multi-valued function if:

$$
\begin{equation}
    \phi (\Delta x_{k})=
	\begin{cases}
		\phi_1, & if \ \Delta x_{k} > \epsilon; \\
		\phi_2, & if \ \Delta x_{k} < \epsilon; \\
		\phi_3, & if \ \Delta x_{k} = \epsilon; \\
	\end{cases}
\end{equation}
\tag{1}
$$

where $\epsilon \in \mathbb{R}$, $\phi_1 \neq \phi_2 \neq \phi_3$. For some inputs  $\Delta x_{k}\neq \epsilon, \ \forall{k} \in \mathbb{N}$ , and the last value in equation above is not used.

A frequently used multi-valued function is the sign$(\cdot): \mathbb{R} \rightarrow \mathbb{R}$:

$$
 \begin{equation}
 sign(x)=
	\begin{cases}
		1, & if \ x > 0; \\
		-1, & if \ x < 0; \\
		0, & if \ x = 0. \\
	\end{cases}
\end{equation}
\tag{2}
$$


### Hysteresis loops in continuous time $\mathcal{H}_t(\omega)$

Let $x_t$ be a continuous-time loading-unloading quasi-static signal applied to a continuous-time system and $y_t$ is the system output. $\mathcal{H}_t(\omega)$ denotes a closed loop in the $x_t - y_t$ plane, which shape depend on $\omega$. If the system presents hysteretic nonlinearity, $\mathcal{H}_t(\omega)$ is denoted as:

$$
\begin{equation}
\mathcal{H}_t(\omega) =
	\begin{cases}
		\mathcal{H}_t(\omega)^{+}, \ for \ t_i \ \leq \ t \ \leq \ t_m, \\
		\mathcal{H}_t(\omega)^{-}; \ for \ t_m \ \leq \ t \ \leq \ t_f, \\
	\end{cases}
\end{equation}
\tag{3}
$$

where $\mathcal{H}_t(\omega)^{+} \neq \mathcal{H}_t(\omega)^{-}$, $\forall t \neq t_m$. $t_i \leq t \leq t_m$ and~$t_m \leq t \leq t_f$ correspond to the regime when $x_t$ is loading and unloading, respectively. $\mathcal{H}_t(\omega)^{+}$ corresponds to the part of the loop formed in the $x_t - y_t$ plane, while $t_i \leq t \leq t_m$ (when $x_t$ is loading) whereas $\mathcal{H}_t(\omega)^{-}$ is the part of the loop formed in the~$x_t - y_t$ plane for~$t_m \leq t \leq t_f$ (when $x_t$ is unloading), as shown in the Figure 2:

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/hysteresis_loop.png?raw=true)
> Figure 2. Example of a hysteresis curve.


> Rate Independent Hysteresis (RIH) ([Visintin, A.]([Differential Models of Hysteresis](https://link.springer.com/book/10.1007/978-3-662-11557-2)))- The hysteresis behavior is called to be rate independent if the path $ABCD$, which depends on pair $x(t), y(t)$, is invariant with respect to any increasing diffeomorphism~$\varphi : [0,T] \rightarrow [0,T]$, i.e.:

$$
\begin{align}
        F(u \ o \ \varphi, y^{0}) = F(u,y^0)\ o \ \varphi & \ em \ [0,T].
\end{align}
\tag{4}
$$

This means that at any instant $t$, $y(t)$ depends only on $u:[0,T] \rightarrow \mathbb{R}$ and on the order in which values have been attained before $t$. In other words, the memory effect is not affected by the frequency of the input.

## Rate Independent Hysteresis  in polynomial NARX model

[Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968) presented the sufficient conditions for NARX model to represent hysteresis. One of the developed concepts in the Bounding Structure $\mathcal{H}$.

> Bounding Structure $\mathcal{H}$ ([Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968)) - Let $\mathcal{H}_t(\omega)$ be the system hysteresis. $\mathcal{H}= \lim_{\omega \to 0} \mathcal{H}_t(\omega)$ is defined as the bounding structure that delimits $\mathcal{H}_t(\omega)$.

Now, consider a polynomial NARX excited by a loading-unloading quasi-static signal. If the model has one real and stable equilibrium point whose location depends on input and loading/unloading regime, the polynomial will exhibit a Rate Independent Hysteresis loop $\mathcal{H}_t(\omega)$ in the $x-y$ plane.

Here is an example. Let $y_k  =  0.8y_{k-1} + 0.4\phi_{k-1} + 0.2x_{k-1}$, where $\phi_{k} = \rm{sign}(\Delta(x_{k}))$ and $x_{k} = sin(\omega k)$ and $\omega$ is the frequency of the input signal $x$. The equilibria of this model is given by:

$$
\begin{equation}
    \overline{y}(\overline{\phi},\overline{x})=
	\begin{cases}
		\frac{0.6+0.2\overline{x}}{1-0.8} \ = 3 \ + \ \overline{x} \ , & for \ loading; \\
		\frac{-0.6+0.2\overline{x}}{1-0.8} \ = -3 \ + \ \overline{x} \ , & for \ unloading; \\
	\end{cases}
\end{equation}
\tag{5}
$$

where $\overline {x}$ is a loading-unloading quasi-static input signal. Since the equilibrium points are asymptotically stable, the output converges to $\mathcal{H}_k (w)$ in the $x-y$ plane. Note that for a constant input value $x ~ = ~ 1 ~ = ~ \overline{x}$, the equilibrium lies in $\overline{y} ~ = ~ 3$ for loading regime and $\overline {y} ~ = ~ -1$ for unloading regime. Analogously, for $\overline {x} ~ = ~ -1$, the equilibrium lies in $\overline {y} ~ = ~ 1$ for loading regime and $\overline {y} ~ = ~ -3$ for unloading regime, as shown in the figure below:

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/bounded_structure_example.png?raw=true)
> Figure 3. Example of a bounding structure $\mathcal{H}$. The black dots are on $\mathcal{H}_{k}(\omega)$ for model $y_k  =  0.8y_{k-1} + 0.4\phi_{k-1} + 0.2x_{k-1}$. The bounding structure $\mathcal{H}$, in red, confines $\mathcal{H}_{k}(\omega)$.}

As can be observed in the Figure 3, in we guarantee the sufficient conditions proposed by [Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968), a NARX model can reproduce a histeretic behavior. Chapter 10 presents a case study of a system with hysteresis.

The following code can be used to reproduce the behavior shown in Figure 3. Change `w` from $1$ to $0.1$ to see how the bounded structure $\mathcal{H}$ converge to the equilibria of the system.

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
w = 1
t = np.arange(0, 60.1, 0.1)
y = np.zeros(len(t))
x = np.sin(w * t)

# Initialize y and fi
fi = np.zeros(len(t))
# Iterate over the time array to calculate y
for k in range(1, len(t)):
    fi[k] = np.sign(x[k] - x[k-1])
    y[k] = 0.8 * y[k-1] + 0.2 * x[k-1] + 0.4 * fi[k-1]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Example')
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/bounded_structure_example_python.png?raw=true)
> Figure 4.  Reproduction of a bounding structure $\mathcal{H}$ using python.
