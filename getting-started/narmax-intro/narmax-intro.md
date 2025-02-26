---
template: overrides/main.html
title: Introduction to NARMAX Models
---

# Introduction

> Author: Wilson Rocha Lacerda Junior

This is the first in a series of publications explaining a little bit about NARMAX[^1] models. I hope the content of these publications will help those who use or would like to use the SysIdentPy library.

> **Looking for more details on NARMAX models?**
> For comprehensive information on models, methods, and a wide range of examples and benchmarks implemented in SysIdentPy, check out our book:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> This book provides in-depth guidance to support your work with SysIdentPy.
>
> You can also explore the [tutorials in the documentation](https://sysidentpy.org/examples/basic_steps/) for practical, hands-on examples.


### System Identification

As I will use the term **Systems Identification** here and there, let me make a brief definition regarding these terms.

<br>
Systems identification is one of the major areas that deals with the modeling of data-based processes. In this context, the term "system" can be interpreted as any set of operations that process one or more inputs and return one or more outputs. Examples include electrical systems, mechanical systems, biological systems, financial systems, chemical systems … literally anything you can relate to input and output data. The electricity demand is part of a system whose inputs can be, for example, quantity of the population, quantity of water in the reservoirs, season, events. The price of a property is the output of a system whose entries can be the city, per capita income, neighborhood, number of rooms, how old the house is, and many others. You got the idea.

<br>
Although there are many things related with Machine Learning, Statistical Learning and other fields,  each field has its particularities.


### So, what is a NARMAX model?

You may have noticed the similarity between the acronym NARMAX with the well-known models ARX, ARMAX, etc., which are widely used for forecasting time series. And this resemblance is not by chance. The Autoregressive models with Moving Average and Exogenous Input (ARMAX) and their variations AR, ARX, ARMA (to name just a few) are one of the most used mathematical representations for identifying linear systems.

<br>
Let's go back to the model. I said that the **ARX** family of models is commonly used to model linear systems. Linear is the key word here. For nonlinear scenarios we have the **NARMAX** class. As reported by Billings (one of the creators of NARMAX model) in the book [Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains],  NARMAX started out as a model name, but soon became a philosophy when it comes to identifying nonlinear systems. Obtaining NARMAX models consists of performing the following steps:

  [^1]:
    Non-linear Autoregressive Models with Moving Average and Exogenous Input.

  [Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains](https://www.wiley.com/en-us/Nonlinear+System+Identification:+NARMAX+Methods+in+the+Time,+Frequency,+and+Spatio+Temporal+Domains-p-9781119943594)

- Dynamical tests and collecting data;
- Choice of mathematical representation;
- Detection of the model structure;
- Estimation of parameters;
- Validation;
- Analysis of the model.

We will cover each of these steps in further publications. The idea of this text is to present an overview of NARMAX models.

<br>
NARMAX models **are not**, however, a simple extension of ARMAX models. NARMAX models are able to represent the most different and complex nonlinear systems. Introduced in 1981 by the Electrical Engineer Stephen A. Billings, NARMAX models can be described as:

$$
    y_k= F^\ell[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
$$

where $n_y\in \mathbb{N}$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$ , are the maximum lags for the system output and input respectively; $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$ is the system output at discrete time $k \in \mathbb{N}^n$; $e_k \in \mathbb{R}^{n_e}$ stands for uncertainties and possible noise at discrete time $k$. In this case, $\mathcal{F}^\ell$ is some nonlinear function of the input and output regressors with nonlinearity degree $\ell \in \mathbb{N}$ and $d$ is a time delay typically set to $d=1$.

If we do not include noise terms, $e_{k-n_e}$, we have NARX models. If we set $\ell = 1$ then we deal with ARMAX models; if $\ell = 1$ and we do not include input and noise terms, it turns to AR model (ARX if we include inputs, ARMA if we include noise terms instead); if $\ell>1$ and there is no input terms, we have the NARMA. If there is no input or noise terms, we have NAR. There are several variants, but that is sufficient for now.

### NARMAX Representation

There are several nonlinear functions representations to approximate the unknown mapping $\mathrm{f}[\cdot]$ in the NARMAX methods, e.g.,

- neural networks;
- fuzzy logic-based models;
- radial basis functions;
- wavelet basis;
- **polynomial basis**;
- generalized additive models;

The remainder of this post contemplates methods related to the power-form polynomial models, which is the most common used representation. Polynomial NARMAX is a mathematical model based on difference equations and relates the current output as a function of past inputs and outputs.

### Polynomial NARMAX

The polynomial NARMAX model with asymptotically stable equilibrium points can be described as:

\begin{align}
    y_k =& \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{j=1}^{q}\Theta_{e}^{j}e_{k-j} + \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m}\\
    &+ \sum_{i=1}^{p}\sum_{j=1}^{q}\Theta_{ye}^{ij}y_{k-i} e_{k-j} + \sum_{i=1}^{p}\sum_{m=1}^{r}\Theta_{yx}^{im}y_{k-i} x_{k-m} \\
    &+ \sum_{j=1}^{q}\sum_{m=1}^{r}\Theta_{e x}^{jm}e_{k-j} x_{k-m} \\
    &+ \sum_{i=1}^{p}\sum_{j=1}^{q}\sum_{m=1}^{r}\Theta_{y e x}^{ijm}y_{k-i} e_{k-j} x_{k-m} \\
    &+ \sum_{m_1=1}^{r} \sum_{m_2=m_1}^{r}\Theta_{x^2}^{m_1 m_2} x_{k-m_1} x_{k-m_2} \dotsc \\
    &+ \sum_{m_1=1}^{r} \dotsc \sum_{m_l=m_{l-1}}^{r} \Theta_{x^l}^{m_1, \dotsc, m_2} x_{k-m_1} x_{k-m_l}
\end{align}

where $\sum\nolimits_{0}$, $c_{y}^{i}$, $c_{e}^{j}$, $c_{x}^{m}$, $c_{y\e}^{ij}$, $c_{yx}^{im}$, $c_{e x}^{jm}$, $c_{y e x}^{ijm}$, $c_{x^2}^{m_1 m_2} \dotsc c_{x^l}^{m_1, \dotsc, ml}$ are constant parameters.

<br>
Let's take a look at an example of a NARMAX model for an easy understanding. The following is a NARMAX model of degree~$2$, identified from experimental data of a DC motor/generator with no prior knowledge of the model form. If you want more information about the identification process, I wrote a paper comparing a polynomial NARMAX with a neural NARX model using that data (IN PORTUGUESE: Identificação de um motor/gerador CC por meio de modelos polinomiais autorregressivos e redes neurais artificiais)

\begin{align}
    y_k =& 1.7813y_{k-1}-0,7962y_{k-2}+0,0339x_{k-1} -0,1597x_{k-1} y_{k-1} +0,0338x_{k-2} \\
    & + 0,1297x_{k-1}y_{k-2} - 0,1396x_{k-2}y_{k-1}+ 0,1086x_{k-2}y_{k-2}+0,0085y_{k-2}^2 + 0.1938e_{k-1}e_{k-2}
\end{align}

But how those terms were selected? How the parameters were estimated? These questions will lead us to model structure selection and parameter estimation topics, but, for now,  let us discuss about those topics in a more simple manner.

<br>
First, the "structure" of a model is the set of terms (also called regressors) included in the final model. The parameters are the values multiplying each of theses terms. And looking at the example above we can notice an really important thing regarding polynomial NARMAX models dealt in this text: they have a non-linear structure, but they are linear-in-the-parameters. You will see how this note is important in the post about parameter estimation.

<br>
In this respect, consider the case where we have the input and output data of some system. For the sake of simplicity, suppose one input and one output. We have the data, but we do not know which lags to choose for the input or the output. Also, we know nothing about the system non-linearity. So, we have to define some values for maximum lags of the input, output and the noise terms, besides the choice of the $\ell$ value. It's worth to notice that many assumptions taken for linear cases are not valid in the nonlinear scenario and therefore select the maximum lags is not straightforward. So, how those values can make the modeling harder?

<br>
So we have one input and one output (disregard the noise terms for now). What if we choose the $n_y = n_x = \ell = 2$? With these values, we have the following possibilities for compose the final model:

\begin{align}
    & constant, y_{k-1}, y_{k-2}, y_{k-1}^2, y_{k-2}^2, x_{k-1}, x_{k-2}, x_{k-1}^2, x_{k-2}^2,y_{k-1}y_{k-2},\\
    & y_{k-1}x_{k-1}, y_{k-1}x_{k-2}, y_{k-2}x_{k-1}, y_{k-2}x_{k-2}, x_{k-1}x_{k-2} .
\end{align}

So we have $15$ candidate terms to compose the final model.

<br>
Again, we do not know how of those terms are significant to compose the model. One should decide to use all the terms because there are only $15$. This, even in a simple scenario like this, can lead to a very wrong representation of the system that you are trying to modeling. Ok, what if we run a brute force algorithm to test the candidate regressors so we can select only the significant ones? In this case, we have $2^{15} = 32768$ possible model structures to be tested.

<br>
You can think that it is ok, we have computer power for that. But this case is very simple and the system might have lags equal to $10$ for input and output. If we define $n_y = n_x = 10$ and $\ell=2$, the number of possible models to be tested increases to $2^{231}=3.4508732\times10^{69}$. If the non-linearity is set to $3$ then we have $2^{1771} = 1.3308291989700907535925992... \times 10^{533}$ candidate models.

<br>
Now, think about the case when we have not 1, but 5, 10 or more inputs... and have to include terms for the noise, and maximum lags are higher than 10... and nonlinearity is higher than 3...

<br>
And the problem is not solved by only identifying the most significant terms. How do you choose the number of terms to include in the final model. It is not just about check the relevance of each regressor, we have to think about the impact of including $5$, $10$ or $50$ regressors in the model. And do not forget: after selecting the terms, we have to estimate its parameters.

<br>
As you can see, to select the most significant terms from a huge dictionary of possible terms is not an easy task. And it is hard not only because the complex combinatorial problem and the uncertainty concerning the model order. Identifying the most significant terms in a nonlinear scenario is very difficult because depends on the type of the non-linearity (sparse singularity or near-singular behavior, memory or dumping effects and many others), dynamical response (spatial-temporal systems, time-dependent), the steady-state response,  frequency of the data, the noise...

<br>
Despite all this complexity, NARMAX models are widely used because it is able to represent complex system with simple and transparent models, which terms are selected using robust algorithms for model structure selection. Model structure selection is the core of NARMAX methods and the scientific community is very active on improving classical methods and developing new ones. As I said, I will introduce some of those methods in another post.

<br>
I hope this publication served as a brief introduction to NARMAX models. Furthermore, I hope I have sparked your interest in this model class. The link to the other texts will be made available soon, but feel free to contact us if you are interested in collaborating with the SysIdentPy library or if you want to address any questions.
