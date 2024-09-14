![](https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/Nonlinear_System_identification.png)

All the world is a nonlinear system

He linearised to the right

He linearised to the left

Till nothing was right

And nothing was left

> [Stephen A. Billings]([Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains | Wiley](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594))



# Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy

Welcome to our companion book on System Identification! This book is a comprehensive approach to learning about dynamic models and forecasting. The main aim of this book is to describe a comprehensive set of algorithms for the identification, forecasting and analysis of nonlinear systems.

Our book is specifically designed for those who are interested in learning system identification and forecasting.  We will guide you through the process step-by-step using Python and the [SysIdentPy](https://github.com/wilsonrljr/sysidentpy) package. With SysIdentPy, you will be able to apply a range of techniques for modeling dynamic systems, making predictions, and exploring different design schemes for dynamic models, from polynomial to neural networks. This book is for graduates, postgraduates, researchers, and for all people from different research areas who have data and want to find models to understand their systems better.

The research literature is filled with books and papers covering various aspects of nonlinear system identification, including NARMAX methods. In this book, our objective isn't to replicate all the numerous algorithm variations available. Instead, we want to show you how to model your data using those algorithms with SysIdentPy. We'll mention all the specific details and different versions of the algorithms in the book, so if you're more interested in the theoretical aspects, you can explore those ideas further. We aim to focus on the fundamental techniques, explaining them in straightforward language and showing how to use them in real-world situations. While there will be some math and technical details involved, the aim is to keep it as easy to understand as possible. In essence, this book aims to be a resource that readers from various fields can use to learn how to model dynamic nonlinear systems.

The best part about our book is that it is open source material, meaning that it is freely available for anyone to use and contribute to. We hope this brings together people who share interest for system identification and forecasting techniques, from linear to nonlinear models.

So, whether you're a student, researcher, data scientist or practitioner, we invite you to share your knowledge and contribute with us. Let’s explore system identification and forecasting with **SysIdentPy**!

To follow along with the Python examples in the book, you’ll need to have some packages installed. We’ll cover the main ones here and let you know if any additional packages are required as we proceed.

```
import sysidentpy
import pandas as pd
import numpy as np
import torch
import matplotlib
import scipy
```

## About the Author

Wilson Rocha is the Head of Data Science at RD Saúde and the creator of the SysIdentPy library. He holds a degree in Electrical Engineering and a Master's in Systems Modeling and Control, both from Federal University of São João del-Rei (UFSJ), Brazil. Wilson began his journey in Machine Learning by developing soccer-playing robots and continues to advance his research in the fields of Multi-objective Nonlinear System Identification and Time Series Forecasting.

Connect with Wilson Rocha through the following social networks:

- [LinkedIn](https://www.linkedin.com/in/wilsonrljr/)
- [ResearchGate](https://www.researchgate.net/profile/Wilson-Lacerda-Junior-2)
- [Discord](https://discord.gg/8eGE3PQ)

## Referencing This Book

If you find this book useful, please cite it as follows:

```
Lacerda Junior, W.R. (2024). *Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*. Web version. https://sysidentpy.org
```

If you use SysIdentPy on your project, please [drop me a line](mailto:wilsonrljr@outlook.com).

If you use SysIdentPy on your scientific publication, we would appreciate citations to the following paper:
- Lacerda et al., (2020). SysIdentPy: A Python package for System Identification using NARMAX models. Journal of Open Source Software, 5(54), 2384, https://doi.org/10.21105/joss.02384

```
@article{Lacerda2020,
  doi = {10.21105/joss.02384},
  url = {https://doi.org/10.21105/joss.02384},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {54},
  pages = {2384},
  author = {Wilson Rocha Lacerda Junior and Luan Pascoal Costa da Andrade and Samuel Carlos Pessoa Oliveira and Samir Angelo Milani Martins},
  title = {SysIdentPy: A Python package for System Identification using NARMAX models},
  journal = {Journal of Open Source Software}
}
```

## PDF, Epub and Mobi version

Download the pdf version of the book: [pdf version](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/Nonlinear_System_Identification_Theory_and_Practice_With_SysIdentPy_Wilson_R_L_Junior.pdf){:download="Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy"}

Download the epub version of the book: [epub version](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/Nonlinear_System_Identification_Theory_and_Practice_With_SysIdentPy_Wilson_R_L_Junior.epub){:download="Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy"}

Download the mobi version of the book: [mobi version](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/Nonlinear_System_Identification_Theory_and_Practice_With_SysIdentPy_Wilson_R_L_Junior.mobi){:download="Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy"}

## Acknowledgments

The System Identification class taught by [Samir Martins](https://ufsj.edu.br/martins/)  (in Portuguese) has been a great source of inspiration for this series. In this book, we will explore Dynamic Systems and learn how to master NARMAX models using Python and the SysIdentPy package. The Stephen A. Billings book, [Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio - Temporal Domains](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594), have been instrumental in showing us how powerful System Identification can be.

In addition to these resources, we will also reference Luis Antônio Aguirre Introdução à [Identificação de Sistemas. Técnicas Lineares e não Lineares Aplicadas a Sistemas. Teoria e Aplicação](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas) (in Portuguese), which has proven to be an invaluable tool in introducing complex dynamic modeling concepts in a straightforward way. As an open source material on System Identification and Forecasting, this book aims to provide a accessible yet rigorous approach to learning dynamic models and forecasting.

## Support the Project

The **Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy** is an extensive open-source resource dedicated to the science of System Identification. Our goal is to make this knowledge accessible to everyone, both financially and intellectually.

If this book has been valuable to you, and you'd like to support our efforts, we welcome financial contributions through our [Sponsor page](https://github.com/sponsors/wilsonrljr).

If you're not in a position to contribute financially, you can still support by helping us improve the book. We encourage you to report any typos, suggest edits, or provide feedback on sections that you found challenging. You can do this by visiting the book's repository and opening an issue. Additionally, if you enjoyed the content, please consider sharing it with others who might benefit from it, and give us a star on [GitHub](https://github.com/wilsonrljr/sysidentpy).

Your support, in any form, helps us continue to enhance this project and maintain a high-quality resource for the community. Thank you for your contribution!
