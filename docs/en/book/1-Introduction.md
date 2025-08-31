# Introduction

The concept of a mathematical model is fundamental in many fields of science. From engineering to sociology, models plays a central role to the study of complex systems as they allow to simulate what will happen in different scenarios and conditions, predict its output for a given input, analyse its properties and explore different design schemes. To accomplish these goals, however, it is crucial that the model is a proper representation of the system under study. The modeling of dynamic and steady-state behaviors is, therefore, fundamental to this type of analysis and depends on System Identification (SI) procedures.

## Models

Mathematical modeling is a great way to understand and analyze different parts of our world. It gives us a clear framework to make sense of complex systems and how they behave. Whether it’s for everyday tasks or big-picture issues like disease control, models are a key part of how we deal with various challenges.

Typing efficiently on a conventional QWERTY keyboard layout is the result of a well-learned model of the QWERTY keyboard embedded in the individual cognitive processes. However, if you are faced with a different keyboard layout, such as the Dvorak or AZERTY, you will probably struggle to adapt to the new model. The system changed, so you will have to update you *model*.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/QWERTY.png?raw=true)
> [QWERTY - Wikipedia](https://en.wikipedia.org/wiki/QWERTY) - [ANSI](https://en.wikipedia.org/wiki/ANSI "ANSI") QWERTY keyboard layout (US)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/AZERTY.png?raw=true)
> [AZERTY](https://en.wikipedia.org/wiki/AZERTY) layout used on a keyboard

Mathematical modeling touches on many parts of our lives. Whether we're looking at economic trends, tracking how diseases spread, or figuring out consumer behavior, models are essential tools for gaining knowledge, making informed decisions, and take control over complex systems.

In essence, mathematical models help us make sense of the world. They let us understand human behavior and the systems we deal with every day. By using these models, we can learn, adapt, and adjust our strategies to keep up with the surrounding changes.

## System Identification

System identification is a data-driven framework to model dynamical systems. Initially, scientists focused on linear system identification, but this has been changing over the past decades with more emphasis in nonlinear systems. Nonlinear system identification is widely considered to be one of the most important topics concerning the modeling of many different dynamical systems, from time-series to severally nonlinear dynamic behaviors.

Extensive resources, including excellent textbooks covering linear system identification and time series forecasting are readily available. In this book, we revisit some known topics, but we also try to approach such subjects in a different and complementary way. We will explore the modeling of nonlinear dynamic systems  using NARMAX(Nonlinear AutoRegressive Moving Average model with eXogenous inputs) methods, which were introduced by [Stephen A. Billings] in [1981](https://pdf.sciencedirectassets.com/314898/1-s2.0-S1474667082X74355/1-s2.0-S1474667017630398/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIAlDD9s3TmWmj7vi6jUiyGu3%2B4wOlhUltouuMtDCf7DdAiBibaBn42D8EkLzeKS6NhEc2E5PPjz%2BpNf7fxe7GuuZ1Cq7BQiW%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMft9sy4HmUQsgz1JTKo8FyQuLbLHlwSW1p7EBeDgywpc0moBrT8CdqIjV2ucAEJ%2Bxf6PZVgMRTE0KPuxx6tNksRk827UBbXjvWD2b7FCIdNMczpYDcD9LPL0xM3SojpHYLUN9nMOVqssnts1C0efyJrowQbn6Jd6LGGHuF3%2BCnaLsxMTyf8pt%2FlLeYyFzLSe8ins0NcC%2BBWR476hcCSY5fbwU2JxQWLFZPv2xAS7WUge0YiMlc3W%2FmZY8Zx3yTgvnQOVm7qwlq7HM9QVc8hoaoMvPmJH0ZzIAxSbqxuRWwCTE712FOW1CQ1upVRksesVdDX3Tv%2BItXKAp%2Blv2ijKpPDn2z8F1zt1Om%2FutokMZzJZZ5w07PtDgkq23px%2BU6CpXlEZXtAyJRyxXChffGK6Ac7QaBt4vMTuHmD8kqJDqEln2qJYZghUn%2FZx0%2B6NaxkpbdV8u5iG2PnHEwn2FHGO1JKIaewUAV%2BXA92v%2FJjVVVkoqLdR4j4BQOSa2%2F69Scc%2BZq5a29zOHX46lXbXtONYoskQP69GJlLHgfEV3tPoDEe0P%2F3r31muBK4a3qeXqnaLS1TzoBjHqEwiDBlhFbxpIsjDhctWxEo84jGDuyjyz8ByvF%2FcRQ6U73Q%2Bre0SmQABoniognhfL40RXVua0si7CASO2I1y6vmQvR4yGUsG2g5%2FizxKZqWuTeJRMIQqrmTjzgK1EOjpn8B4og68x8hcxVGzd1Bb0i%2FZ0HsXyBZ2DVG7YykR6I8NGQk7pNFeF3PcF5r446wc8vgVYvDy4yq1GkyaGKsI32TQSnYk5aDKOo%2Fx05HHhE1juw9bROiKYrJV%2BDmUj1ToH3AT1YOW4U%2FYyoGoLl2q0QdUi9zRW%2FA%2FCaWIZ3XYtB%2BY7xEFa7YZiSjCmlLW1BjqyAZDOCbxeC9NMZd8ZbHQNkV1tGJLpsuFvFKwPplQw4w3ZFd1F0KEcbH5NqECMYDqFER%2Bnvlpxl%2FYoBODHrVfxUvM7bv3PL5Jhdb8DdaoygIUFgMADVdeGquP0F7FYUeXbJ1s0kJSO5TkXvkyEalxA7Hg1DpvpTXhE7Had9exiuzBPC4A7pLISpdguQkKVbt4gxarxiLPihb5rpueXLp4g2rTcLgls%2BHr5jO4C7wQRbY8QI48%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240802T220345Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3JGFOCJZ%2F20240802%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=82ab64cae9cae307cbfaf00a3ab07182268c43e6a05052f6172128e80bd86e65&hash=ba5f80609dd1f2a175c54cebdbf1c92094ee30786c7ffd01a10468d28f059112&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1474667017630398&tid=spdf-0f070471-6549-4439-ba88-fc62fb637581&sid=e715fa3e9ec8c847157af269cf9c0ee0d69cgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=181658050b5351005659&rr=8ad15c165cd77e01&cc=br).

## Linear or Nonlinear System Identification

### Linear Models

While most real world systems are nonlinear, you probably should give linear models a try first. Linear models usually serves as a strong baseline and can be good enough for your case, giving satisfactory performance. [Astron and Murray](https://www.cds.caltech.edu/~murray/books/AM05/pdf/am06-complete_16Sep06.pdf) and [Glad and Ljung](https://www.taylorfrancis.com/books/mono/10.1201/9781315274737/control-theory-lennart-ljung-torkel-glad) showed that many nonlinear systems can be well described by locally linear models. Besides, linear models are easy to fit, easy to interpret, and requires less computational resources than nonlinear models, allowing you to experiment fast and gather insights before thinking about gray box models or complex nonlinear models.

Linear models can be very useful, even in the presence of strong nonlinearities, because it is much easier to deal with it. Moreover, the development of linear identification algorithms is still a very active and healthy research field, with many papers being released every year [Sai Li, Linjun Zhang, T. Tony Cai & Hongzhe Li](https://www.tandfonline.com/doi/abs/10.1080/01621459.2023.2184373), [Maria Jaenada, Leandro Pardo](https://www.mdpi.com/1099-4300/24/1/123), [Xing Liu; Lin Qiu, Youtong Fang; Kui Wang; Yongdong Li, Jose Rodríguez](https://ieeexplore.ieee.org/abstract/document/10296948), [Alessandro D’Innocenzo and Francesco Smarra](https://www.paperhost.org/proceedings/controls/ECC24/files/0026.pdf). Linear models work well most of the time and should be the first choice for many applications. However, when dealing with complex systems where linear assumptions don’t hold, nonlinear models become essential.

### Nonlinear Models

When linear models do not perform well enough, you should consider nonlinear models. It's important to notice, however, that changing from a linear to a nonlinear model is not always a simple task. For inexperienced users, it's common to build nonlinear models that performs worse than the linear ones. To work with nonlinear models, you must consider that characteristics such structural errors, noise, operation point, excitation signals and many others aspects of your system under study impact your modelling approach and strategy.

> As suggested by Johan Schoukens and Lennart Ljung in "[Nonlinear System Identiﬁcation - A User-Oriented Roadmap](https://arxiv.org/pdf/1902.00683)", only start working with nonlinear models if there is enough evidence that linear models will not solve the problem.

Nonlinear models are more flexible than linear models and can be built using many different mathematical representations, such as polynomial, generalized additive, neural networks, wavelet and many more ([Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594)). Such flexibility, however, makes nonlinear system identification much more complex the linear ones, from the experiment design to the model selection. The user should consider that, besides the modeling complexity, moving to nonlinear models will require a revision in the road-map and computational resources defined when dealing with the linear models. In this respect, always ask yourself whether the potential benefits of nonlinear models are worth the effort.

## NARMAX Methods

NARMAX model is one of the most frequently employed nonlinear model representation and is widely used to represent a broad class of nonlinear systems. NARMAX methods were successfully applied in many scenarios, which include industrial processes, control systems, structural systems, economic and financial systems, biology, medicine, social systems, and much more. The NARMAX model representation and the class of systems that can be represented by it will be discussed later in the book.

The main steps involved to build NARMAX models are [(Billings, 2013)](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594):

1. Model Representation: define the model mathematical representation.
2. Model Structure Selection: define which terms are in the final model.
3. Parameter Estimation: estimate the coefficients of each model term selected in step 1.
4. Model validation: to make sure the model is unbiased and accurate;
5. Model Prediction/Simulation: predict future outputs or simulate the behavior of the system given different inputs.
6. Analysis: understanding the dynamical properties of the system under study
7. Control: develop control design schemes based on the obtained model.

Model Structure Selection (MSS) is the most important aspect of NARMAX methods and the most complex. Selecting the model terms is fundamental if the goal of the identification is to obtain models that can reproduce the dynamics of the original system and impacts every other aspect of the identification process. Problems related to over parameterization and numerical ill-conditioning are typical because of the limitations the identification algorithms in selecting the appropriate terms that should compose the final model ([L. A. Aguirre e S. A. Billings](https://www.sciencedirect.com/science/article/abs/pii/0167278995900535), [L. Piroddi e W. Spinelli](https://www.tandfonline.com/doi/abs/10.1080/00207170310001635419)).

> In SysIdentPy, you are allowed to interact directly with every item described in the 7 steps, except for the control one. SysIdentPy focuses on modeling, not on control design. You'll have to use some of the code bellow in every modeling task using SysIdentPy. You'll learn the details along the book, so don't worry if you are not familiar with those methods yet.

```
from sysidentpy.basis_function import Polynomial
from sysidentpy.neural_network import NARXNN
from sysidentpy.general_estimators import NARX
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import RecursiveLeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.simulation import SimulateNARMAX

```

## What is the Purpose of System Identification?

Because of the Model Structure Selection problem, [Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594) states that the goal of System Identification using NARMAX  methods is twofold: performance and parsimony.

The first goal is often about approximation. Here, the main focus is to build a model that make predictions with with the lowest error possible. This approach is common in applications like weather forecasting, demand forecasting, predicting stock prices, speech recognition, target tracking, and pattern classification. In these cases, the specific form of the model isn't as critical. In other words, how the terms interact (in parametric models), the mathematical representation, the static behavior and so on are not that important; what matters most is finding a way to minimize prediction errors.

But system identification isn't just about minimizing prediction errors. One of the main goals of System Identification is to build models that help the user to understand and interpret the system being modeled. Beyond just making accurate predictions, the goal is to develop models that truly capture the dynamic behavior of the system being studied, ideally in the simplest form possible. Science and engineering are all about understanding systems by breaking down complex behaviors into simpler ones that we can understand and control. For example, if the system's behavior can be described by a simple first-order dynamic model with a cubic nonlinear term in the input, system identification should help uncover that.

## Is System Identification Machine Learning?

First, let's take an overview of static and dynamic systems. Imagine you have an electric guitar connected to an effect processor that can apply various audio effects, such as reverb or distortion. The effect is controlled by a switch that toggles between "on" and "off". Let’s consider this from the perspective of signals. The input signal represents the state of the effect switch: switch off (low level), switch on (high level). If we represent the guitar signal, we have a binary condition: effect off (original guitar sound), effect on (modified guitar sound). This is an example of a static system: the output (guitar sound) directly follows the input (state of the effect switch).

When the effect switch is off, the output is just the clean, unaltered guitar signal. When the effect switch is on, the output is the guitar signal with the effect applied, such as amplification or distortion. In this system, the effect being on or off directly influences the guitar signal without any delay or additional processing.

This example illustrates how a static system operates with binary control inputs, where the output directly reflects the input state, providing a straightforward mapping between the control signal and the system’s response.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

t = np.linspace(0, 30, 500, endpoint=False)
u = signal.square(0.2*np.pi * t)
u[u < 0] = 0
# In a static system, the output y directly follows the input u
y = u

# Plot the input and output
plt.figure(figsize=(15, 3))
plt.plot(t, u, label='Input (State of the Switch)', color="grey", linewidth=10, alpha=0.5)
plt.plot(t, y, label='Output (Static System Response)', color='k', linewidth=0.5)
plt.title('Static System Response to the Input')
plt.xlabel('Time [s]')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/static_example.png?raw=true)
> Static response representation. The input signal representing the state of the switch (switch off (low level), switch on (high level)), and the static response: original sound (low level), processed sound (high level).

Now, let’s consider a dynamic system: using an air conditioner to lower the room temperature. This example effectively illustrates the concepts of dynamic systems and how their output responds over time.

Let’s imagine this from the perspective of signals. The input signal represents the state of the air conditioner's control: turning the air conditioner on (high level) or turning it off (low level). When the air conditioner is turned on, it begins to cool the room. However, the room temperature does not drop instantaneously to the desired cooler level. It takes time for the air conditioner to affect the temperature, and the rate at which the temperature decreases can vary based on factors like the room size and insulation.

Conversely, when the air conditioner is turned off, the room temperature does not immediately return to its original ambient temperature. Instead, it gradually warms up as the cooling effect diminishes.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/dynamic_example.png?raw=true)
> Using an air conditioner to lower the room temperature as dynamic system representation.

In this dynamic system, the output (room temperature) does not instantly follow the input (state of the air conditioner) because there is a time lag involved in both cooling and warming processes. The system has memory, meaning the current room temperature depends not only on the current state of the air conditioner but also on how long it has been running or off, and how much it has already cooled or allowed the room to warm up.

This example highlights the nature of dynamic systems: the response to an input is gradual and affected by the system’s internal dynamics. The air conditioner’s effect on the room temperature exemplifies how dynamic systems have a time-dependent response, where the output changes over time and does not immediately match the input signal.

For static systems, the output is a direct function of the input, represented by an algebraic equation:

$$
y(t) = G \cdot u(t)
$$

For dynamic systems, the output depends on the input and the rate of change of the input, represented by a differential equation. For example, the output $y(t)$ can be modeled as:

$$
y(t) = G \cdot u(t) - \tau \cdot \frac{dy(t)}{dt}
$$

Here, $G$ is the gain, and $\tau$ is a constant that incorporates the system's memory. For discrete-time systems, we consider signals at specific and spaced intervals. The differential equation is discretized, and the derivative is approximated by a finite difference:

$$
y[k] = \alpha y[k-1] + \beta u[k]
$$

where $\alpha$ and $\beta$ are constants that determine the system's response. The z-transform can be used to obtain the transfer function in the z-domain.

In summary, static systems are modeled by algebraic equations, while dynamic systems are modeled by differential equations.

> As Luis Antonio Aguirre states in one of his [classes on YouTube (in Portuguese)](https://www.youtube.com/watch?v=OVs0p2jem1Q), **all physical systems are dynamic, but depending on the timescale, they can be modeled as static for simplification**. For example, the transition between the effects on the guitar sound, if taken in seconds (as we did in the example), could be treated as static depending on your analysis. However, the pedal board have components like capacitors, which are dynamic electrical components, making it a dynamic system. The response, however, is so fast that we dealt with it like a static system. Therefore, representing a system as static is a **modeling decision**.

Table 1 shows how this field can be categorized with respect to linear/nonlinear and static/dynamic systems.

| System Characteristics | Linear Model                 | Nonlinear Model                 |
| ---------------------- | ---------------------------- | ------------------------------- |
| Static                 | Linear Regression            | Machine Learning                |
| Dynamic                | Linear System Identification | Nonlinear System Identification |
> Table 1: Naming conventions in the System Identification field. Adapted from [Oliver Nelles](https://link.springer.com/book/10.1007/978-3-662-04323-3#author-0-0)

## Nonlinear System Identification and Forecasting Applications: Case Studies

There’s a lot of research out there on nonlinear system identification, including NARMAX methods. However, there are a relatively small number of books and papers showing how to apply these methods to real-life systems Instead in a way that’s easy to understand. Our goal with this book is to change that. We want to make these methods practical and accessible. While we’ll cover the necessary math and algorithms, we’ll keep things as clear and simple as possible, making it easier for readers from all backgrounds to learn how to model dynamic nonlinear systems using **SysIdentPy**.

Therefore, this book aims to fill a gap in the existing literature. In Chapter 10, we present real-world case studies to show how NARMAX methods can be applied to a variety of complex systems. Whether it’s modeling a highly nonlinear system like the Bouc-Wen model, modeling a dynamic behavior in a full-scale F-16 aircraft, or working with the M4 dataset for benchmarking, we’ll guide you through building NARMAX models using **SysIdentPy**.

The case studies we've selected come from a wide range of fields, not just the typical timeseries or industrial examples you might expect from traditional system identification or timeseries books. Our aim is to showcase the versatility of NARMAX algorithms and **SysIdentPy** and illustrate the kind of in-depth analysis you can achieve with these tools.

## Abbreviations

| Abbreviation | Full Name                                                    |
| ------------ | ------------------------------------------------------------ |
| AIC          | Akaike Information Criterion                                 |
| AICC         | Corrected Akaike Information Criterion                       |
| AOLS         | Accelerated Orthogonal Least Squares                         |
| ANN          | Artificial Neural Network                                    |
| AR           | AutoRegressive                                               |
| ARMAX        | AutoRegressive Moving Average with eXogenous Input           |
| ARARX        | AutoRegressive AutoRegressive with eXogenous Input           |
| ARX          | AutoRegressive with eXogenous Input                          |
| BIC          | Bayesian Information Criterion                               |
| ELS          | Extended Least Squares                                       |
| ER           | Entropic Regression                                          |
| ERR          | Error Reduction Ratio                                        |
| FIR          | Finite Impulse Response                                      |
| FPE          | Final Prediction Error                                       |
| FROLS        | Forward Regression Orthogonal Least Squares                  |
| GLS          | Generalized Least Squares                                    |
| LMS          | Least Mean Square                                            |
| LS           | Least Squares                                                |
| LSTM         | Long Short-Term Memory                                       |
| MA           | Moving Average                                               |
| MetaMSS      | Meta Model Structure Selection                               |
| MIMO         | Multiple Input Multiple Output                               |
| MISO         | Multiple Input Single Output                                 |
| MLP          | Multilayer Perceptron                                        |
| MSE          | Mean Squared Error                                           |
| MSS          | Model Structure Selection                                    |
| NARMAX       | Nonlinear AutoRegressive Moving Average with eXogenous Input |
| NARX         | Nonlinear AutoRegressive with eXogenous Input                |
| NFIR         | Nonlinear Finite Impulse Response                            |
| NIIR         | Nonlinear Infinite Impulse Response                          |
| NLS          | Nonlinear Least Squares                                      |
| NN           | Neural Network                                               |
| OBF          | Orthonormal Basis Function                                   |
| OE           | Output Error                                                 |
| OLS          | Orthogonal Least Squares                                     |
| RBF          | Radial Basis Function                                        |
| RELS         | Recursive Extended Least Squares                             |
| RLS          | Recursive Least Squares                                      |
| RMSE         | Root Mean Squared Error                                      |
| SI           | System Identification                                        |
| SISO         | Single Input Single Output                                   |
| SVD          | Singular Value Decomposition                                 |
| WLS          | Weighted Least Squares                                       |

## Variables

| Variable Name           | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| $f(\cdot)$              | function to be approximated                                   |
| $k$                     | discrete time                                                 |
| $m$                     | dynamic order                                                 |
| $x$                     | system inputs                                                 |
| $y$                     | system output                                                 |
| $\hat{y}$               | model predicted output                                        |
| $\lambda$               | regularization strength                                       |
| $\sigma$                | standard deviation                                            |
| $\theta$                | parameter vector                                              |
| $N$                     | number of data points                                         |
| $\Psi(\cdot)$           | Information Matrix                                            |
| $n_{m^r}$               | Number of potential regressors for MIMO models                |
| $\mathcal{F}$           | Arbitrary mathematical representation                         |
| $\Omega_{y^p x^m}$      | Term cluster of polynomial NARX                               |
| $\ell$                  | nonlinearity degree of the model                              |
| $\hat{\Theta}$          | Estimated Parameter Vector                                    |
| $\hat{y}_k$             | model predicted output at discrete time $k$                   |
| $\mathbf{X}_k$          | Column vector of multiple system inputs at discrete-time $k$  |
| $\mathbf{Y}_k$          | Column vector of multiple system outputs at discrete-time $k$ |
| $\mathcal{H}_t(\omega)$ | Hysteresis loop of the system in continuous-time              |
| $\mathcal{H}$           | Bounding structure that delimits the system hysteresis loop   |
| $\rho$                  | Tolerance value                                               |
| $\sum_{y^p x^m}$        | Cluster coefficients of polynomial NARX                       |
| $e_k$                   | error vector at discrete-time $k$                             |
| $n_r$                   | Number of potential regressors for SISO models                |
| $n_x$                   | maximum lag of the input regressor                            |
| $n_y$                   | maximum lag of the output regressor                           |
| $n$                     | number of observations in a sample                            |
| $x_k$                   | system input at discrete-time $k$                             |
| $y_k$                   | system output at discrete-time $k$                            |

## Book Organization

This book focuses on making concepts easy to understand, emphasizing clear explanations and practical connections between different methods. We avoid excessive formalism and complex equations, opting instead to illustrate core ideas with plenty of hands-on examples. Written with a System Identification perspective, the book offers practical implementation details throughout the chapters.

The goals of this book are to help you:

- Understand the advantages, drawbacks, and areas of application of different NARMAX models and algorithms.
- Choose the right approach for your specific problem.
- Adjust all hyperparameters properly.
- Interpret and comprehend the obtained results.
- valuate the reliability and limitations of your models.

Many chapters include real-world examples and data, guiding you on how to apply these methods using SysIdentPy in practice.

