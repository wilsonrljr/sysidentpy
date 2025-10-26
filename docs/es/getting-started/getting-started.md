---
template: overrides/main.html
title: Primeros Pasos
---

# Primeros Pasos

Â¡Bienvenido a la documentaciÃ³n de SysIdentPy! Aprende cÃ³mo empezar a usar SysIdentPy en tu proyecto. Luego explora los conceptos principales y descubre recursos adicionales para modelar sistemas dinÃ¡micos y series temporales.

<div class="custom-collapsible-card">
    <input type="checkbox" id="toggle-info">
    <label for="toggle-info">
        ðŸ“š <strong>Â¿Buscas mÃ¡s detalles sobre modelos NARMAX?</strong>
        <span class="arrow">â–¼</span>
    </label>
    <div class="collapsible-content">
        <p>
            Para informaciÃ³n completa sobre modelos, mÃ©todos y un conjunto de ejemplos y benchmarks implementados en <strong>SysIdentPy</strong>, consulta nuestro libro:
        </p>
        <a href="https://sysidentpy.org/book/0-Preface/" target="_blank">
            <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
        </a>
        <p>
            Este libro ofrece una guÃ­a detallada para ayudarte en tu trabajo con <strong>SysIdentPy</strong>.
        </p>
        <p>
            ðŸ› ï¸ TambiÃ©n puedes explorar los <a href="https://sysidentpy.org/user-guide/overview/" target="_blank"><strong>tutoriales en la documentaciÃ³n</strong></a> para ejemplos prÃ¡cticos.
        </p>
    </div>
</div>

## Â¿QuÃ© es SysIdentPy?

SysIdentPy es una biblioteca Python de cÃ³digo abierto para identificaciÃ³n de sistemas usando modelos **NARMAX**, construida sobre **NumPy** y distribuida bajo la licencia BSD de 3 clÃ¡usulas. SysIdentPy proporciona una estructura flexible y fÃ¡cil de usar para construir modelos dinÃ¡micos no lineales para series temporales y sistemas dinÃ¡micos.

Con **SysIdentPy**, puedes:

- Construir y personalizar modelos no lineales para predicciÃ³n de series temporales y sistemas dinÃ¡micos.
- Utilizar tÃ©cnicas innovadoras para selecciÃ³n de estructura y estimaciÃ³n de parÃ¡metros del modelo.
- Experimentar modelos NARX neuronales y otros algoritmos avanzados.

## InstalaciÃ³n

SysIdentPy se publica como un [paquete Python] y puede instalarse con `pip`, preferiblemente en un [entorno virtual]. Si no tienes experiencia, desplaza la pÃ¡gina y amplÃ­a la caja de ayuda. Instala con:

<div class="custom-card">
    <div class="tab-container">
        <!-- Latest Tab -->
        <input type="radio" id="tab-latest" name="tab-group" checked>
        <label for="tab-latest">Ãšltima VersiÃ³n</label>
        <div class="tab-content">
            <pre><code>pip install sysidentpy</code></pre>
        </div>

        <!-- Neural NARX Support Tab -->
        <input type="radio" id="tab-neural" name="tab-group">
        <label for="tab-neural">Soporte NARX Neural</label>
        <div class="tab-content">
            <pre><code>pip install sysidentpy["all"]</code></pre>
        </div>

        <!-- Version x.y.z Tab -->
        <input type="radio" id="tab-version" name="tab-group">
        <label for="tab-version">VersiÃ³n EspecÃ­fica</label>
        <div class="tab-content">
            <pre><code>pip install sysidentpy=="0.5.3"</code></pre>
        </div>

        <!-- Development Versions -->
        <input type="radio" id="tab-git" name="tab-group">
        <label for="tab-git">Desde Git</label>
        <div class="tab-content">
            <pre><code>pip install git+https://github.com/wilsonrljr/sysidentpy.git</code></pre>
        </div>
    </div>

</div>

<div class="custom-collapsible-card">
    <input type="checkbox" id="toggle-dependencies">
    <label for="toggle-dependencies">
        â“ <strong>Â¿CÃ³mo gestionar las dependencias de mi proyecto?</strong>
        <span class="arrow">â–¼</span>
    </label>
    <div class="collapsible-content">
        <p>
            Si no tienes experiencia previa con Python, recomendamos la lectura de
            <a href="https://pip.pypa.io/en/stable/user_guide/" target="_blank">
                Using Python's pip to Manage Your Projects' Dependencies
            </a>, que es una excelente introducciÃ³n a la gestiÃ³n de paquetes en Python y ayuda en la resoluciÃ³n de errores.
        </p>
    </div>
</div>

[paquete Python]: https://pypi.org/project/sysidentpy/
[entorno virtual]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment

## Â¿CuÃ¡les son las principales funcionalidades de SysIdentPy?

<div class="feature-grid">
    <div class="feature-card">
        <a href="https://sysidentpy.org/es/getting-started/quickstart-guide/#model-classes" class="feature-link">
            <h3>ðŸ§© FilosofÃ­a NARMAX</h3>
        </a>
        <p>Construye variaciones como <strong>NARX</strong>, <strong>NAR</strong>, <strong>ARMA</strong>, <strong>NFIR</strong> y otras.</p>
    </div>
    <div class="feature-card">
        <a href="https://sysidentpy.org/es/getting-started/quickstart-guide/#model-structure-selection-algorithms" class="feature-link">
            <h3>ðŸ“ SelecciÃ³n de la Estructura</h3>
        </a>
        <p>Usa mÃ©todos como <strong>FROLS</strong>, <strong>MetaMSS</strong> y combinaciones con tÃ©cnicas de estimaciÃ³n de parÃ¡metros.</p>
    </div>
    <div class="feature-card">
        <a href="https://sysidentpy.org/user-guide/tutorials/basis-function-overview/" class="feature-link">
            <h3>ðŸ”— Funciones Base</h3>
        </a>
        <p>Elige entre <strong>8+ funciones base</strong>, combinando tipos lineales y no lineales para modelos NARMAX personalizados.</p>
    </div>
    <div class="feature-card">
        <a href="https://sysidentpy.org/user-guide/tutorials/parameter-estimation-overview/" class="feature-link">
            <h3>ðŸŽ¯ EstimaciÃ³n de ParÃ¡metros</h3>
        </a>
        <p>MÃ¡s de <strong>15 mÃ©todos</strong> para explorar diferentes escenarios junto con tÃ©cnicas de selecciÃ³n de estructura.</p>
    </div>
    <div class="feature-card">
        <a href="https://sysidentpy.org/user-guide/tutorials/multiobjective-parameter-estimation-overview/" class="feature-link">
            <h3>âš–ï¸ TÃ©cnicas Multiobjetivo</h3>
        </a>
        <p>Minimiza diferentes funciones objetivo usando <strong>informaciÃ³n afÃ­n</strong> para estimaciÃ³n de parÃ¡metros.</p>
    </div>
    <div class="feature-card">
        <a href="https://sysidentpy.org/user-guide/how-to/simulating-existing-models/" class="feature-link">
            <h3>ðŸ”„ SimulaciÃ³n de Modelos</h3>
        </a>
        <p>Reproduce resultados de artÃ­culos con <strong>SimulateNARMAX</strong>. Prueba y compara modelos publicados en artÃ­culos.</p>
    </div>
    <div class="feature-card">
        <a href="https://sysidentpy.org/user-guide/how-to/create-a-narx-neural-network/" class="feature-link">
            <h3>ðŸ¤– NARX Neural (PyTorch)</h3>
        </a>
        <p>Integra con <strong>PyTorch</strong> para arquitecturas NARX neuronales usando cualquier optimizador y funciÃ³n de coste.</p>
    </div>
    <div class="feature-card">
        <a href="https://sysidentpy.org/user-guide/tutorials/general-NARX-models/" class="feature-link">
            <h3>ðŸ› ï¸ Estimadores Generales</h3>
        </a>
        <p>Compatible con <strong>scikit-learn</strong>, <strong>CatBoost</strong> y mÃ¡s para crear modelos NARMAX.</p>
    </div>
</div>

## Recursos adicionales

<ul class="custom-link-list">
    <li>
        <a href="https://sysidentpy.org/developer-guide/contribute/" target="_blank">ðŸ¤ Contribuye con SysIdentPy</a>
    </li>
    <li>
        <a href="https://sysidentpy.org/getting-started/license/" target="_blank">ðŸ“œ InformaciÃ³n de Licencia</a>
    </li>
    <li>
        <a href="https://sysidentpy.org/community-support/get-help/" target="_blank">ðŸ†˜ Ayuda & Soporte</a>
    </li>
    <li>
        <a href="https://sysidentpy.org/community-support/meetups/ai-networks-meetup/" target="_blank">ðŸ“… Charlas</a>
    </li>
    <li>
        <a href="https://sysidentpy.org/landing-page/sponsor/" target="_blank">ðŸ’– Hazte Patrocinador</a>
    </li>
    <li>
        <a href="https://sysidentpy.org/user-guide/API/narmax-base/" target="_blank">ðŸ§© Explora el CÃ³digo Fuente</a>
    </li>
</ul>

## Â¿Te gusta **SysIdentPy**?

Â¿Te gustarÃ­a ayudar a SysIdentPy, a otros usuarios y al autor de la biblioteca? Puedes Â«estrellarÂ» el proyecto en GitHub haciendo clic en el botÃ³n de estrella en la esquina superior derecha de la pÃ¡gina: <a href="https://github.com/wilsonrljr/sysidentpy" class="external-link" target="_blank">https://github.com/wilsonrljr/sysidentpy</a>. â­ï¸

Al marcar un repositorio con estrella, lo encontrarÃ¡s mÃ¡s fÃ¡cilmente en el futuro, recibirÃ¡s sugerencias de proyectos relacionados en GitHub y ademÃ¡s valoras el trabajo del mantenedor.

Considera tambiÃ©n apoyar el proyecto haciÃ©ndote sponsor. Tu apoyo ayuda a mantener el desarrollo activo y garantiza la evoluciÃ³n continua de <strong>SysIdentPy</strong>.

[:octicons-star-fill-24:{ .mdx-heart } &nbsp; SÃ© un <span class="mdx-sponsorship-count" data-mdx-component="sponsorship-count"></span> Patrocinador en GitHub][wilsonrljr's sponsor profile]{ .md-button .md-button--primary .mdx-sponsorship-button }

