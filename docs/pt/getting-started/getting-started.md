---
template: overrides/main.html
title: Primeiros Passos
---

# Primeiros Passos

Bem-vindo à documentação do SysIdentPy! Aprenda como começar a usar o SysIdentPy no seu projeto. Em seguida, explore os principais conceitos e descubra recursos adicionais para modelar sistemas dinâmicos e séries temporais.

<div class="custom-collapsible-card">
	<input type="checkbox" id="toggle-info">
	<label for="toggle-info">
		📚 <strong>Em busca de mais detalhes sobre modelos NARMAX?</strong>
		<span class="arrow">▼</span>
	</label>
	<div class="collapsible-content">
		<p>
			Para informações completas sobre modelos, métodos e um conjunto de exemplos e benchmarks implementados no <strong>SysIdentPy</strong>, confira nosso livro:
		</p>
		<a href="https://sysidentpy.org/book/0-Preface/" target="_blank">
			<em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
		</a>
		<p>
			Esse livro oferece uma orientação detalhada para auxiliar no seu trabalho com o <strong>SysIdentPy</strong>.
		</p>
		<p>
			🛠️ Você também pode explorar os <a href="https://sysidentpy.org/user-guide/overview/" target="_blank"><strong>tutoriais na documentação</strong></a> para exemplos práticos.
		</p>
	</div>
</div>

## O que é o SysIdentPy

SysIdentPy é uma biblioteca Python de código aberto para Identificação de Sistemas usando modelos **NARMAX**, construída sobre o **NumPy** e distribuída sob a licença BSD de 3 cláusulas. SysIdentPy disponibiliza uma estrutura flexível e fácil de usar para construir modelos dinâmicos não lineares para séries temporais e sistemas dinâmicos.

Com o **SysIdentPy**, você pode:

- Construir e customizar modelos não lineares para previsão de séries temporais e sistemas dinâmicos.
- Utilizar técnicas inovadoras para seleção de estrutura e estimação de parâmetros do modelo.
- Experimentar modelos NARX neurais e outros algoritmos avançados.

## Instalação

SysIdentPy é publicado como um [pacote Python] e pode ser instalado com `pip`, de preferência em um [ambiente virtual]. Caso não tenha experiência, role a página e expanda a caixa de ajuda. Instale com:

<div class="custom-card">
	<div class="tab-container">
		<!-- Latest Tab -->
		<input type="radio" id="tab-latest" name="tab-group" checked>
		<label for="tab-latest">Última Versão</label>
		<div class="tab-content">
			<pre><code>pip install sysidentpy</code></pre>
		</div>

		<!-- Neural NARX Support Tab -->
		<input type="radio" id="tab-neural" name="tab-group">
		<label for="tab-neural">Suporte NARX Neural</label>
		<div class="tab-content">
			<pre><code>pip install sysidentpy["all"]</code></pre>
		</div>

		<!-- Version x.y.z Tab -->
		<input type="radio" id="tab-version" name="tab-group">
		<label for="tab-version">Versão Específica</label>
		<div class="tab-content">
			<pre><code>pip install sysidentpy=="0.5.3"</code></pre>
		</div>

		<!-- Versões de Desenvolvimento -->
		<input type="radio" id="tab-git" name="tab-group">
		<label for="tab-git">Do Git</label>
		<div class="tab-content">
			<pre><code>pip install git+https://github.com/wilsonrljr/sysidentpy.git</code></pre>
		</div>
	</div>
</div>

<div class="custom-collapsible-card">
	<input type="checkbox" id="toggle-dependencies">
	<label for="toggle-dependencies">
		❓ <strong>Como gerenciar as dependências do meu projeto?</strong>
		<span class="arrow">▼</span>
	</label>
	<div class="collapsible-content">
		<p>
			Se você não tem experiência prévia com Python, recomendamos a leitura de
			<a href="https://pip.pypa.io/en/stable/user_guide/" target="_blank">
				Using Python's pip to Manage Your Projects' Dependencies
			</a>, que é uma excelente introdução à mecânica de gerenciamento de pacotes em Python e ajuda na solução de erros.
		</p>
	</div>
</div>


  [pacote Python]: https://pypi.org/project/sysidentpy/
  [ambiente virtual]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment

## Quais são os principais recursos do SysIdentPy?

<div class="feature-grid">
	<div class="feature-card">
		<a href="https://sysidentpy.org/pt/getting-started/quickstart-guide/#model-classes" class="feature-link">
			<h3>🧩 Filosofia NARMAX</h3>
		</a>
		<p>Construa variações como <strong>NARX</strong>, <strong>NAR</strong>, <strong>ARMA</strong>, <strong>NFIR</strong> e outras.</p>
	</div>
	<div class="feature-card">
		<a href="https://sysidentpy.org/pt/getting-started/quickstart-guide/#model-structure-selection-algorithms" class="feature-link">
			<h3>📝 Seleção da Estrutura</h3>
		</a>
		<p>Use métodos como <strong>FROLS</strong>, <strong>MetaMSS</strong> e combinações com técnicas de estimação de parâmetros.</p>
	</div>
	<div class="feature-card">
		<a href="https://sysidentpy.org/user-guide/tutorials/basis-function-overview/" class="feature-link">
			<h3>🔗 Funções Base</h3>
		</a>
		<p>Escolha entre <strong>8+ funções base</strong>, combinando tipos lineares e não lineares para modelos NARMAX personalizados.</p>
	</div>
	<div class="feature-card">
		<a href="https://sysidentpy.org/user-guide/tutorials/parameter-estimation-overview/" class="feature-link">
			<h3>🎯 Estimação de Parâmetros</h3>
		</a>
		<p>Mais de <strong>15 métodos</strong> para explorar diferentes cenários em conjunto com técnicas de seleção de estrutura.</p>
	</div>
	<div class="feature-card">
		<a href="https://sysidentpy.org/user-guide/tutorials/multiobjective-parameter-estimation-overview/" class="feature-link">
			<h3>⚖️ Técnicas Multiobjetivo</h3>
		</a>
		<p>Minimize diferentes funções objetivo usando <strong>informação afim</strong> para estimação de parâmetros.</p>
	</div>
	<div class="feature-card">
		<a href="https://sysidentpy.org/user-guide/how-to/simulating-existing-models/" class="feature-link">
			<h3>🔄 Simulação de Modelos</h3>
		</a>
		<p>Reproduza resultados de artigos com <strong>SimulateNARMAX</strong>. Teste e compare modelos publicados em artigos.</p>
	</div>
	<div class="feature-card">
		<a href="https://sysidentpy.org/user-guide/how-to/create-a-narx-neural-network/" class="feature-link">
			<h3>🤖 NARX Neural (PyTorch)</h3>
		</a>
		<p>Integre com <strong>PyTorch</strong> para arquiteturas NARX neurais usando qualquer otimizador e função de custo.</p>
	</div>
	<div class="feature-card">
		<a href="https://sysidentpy.org/user-guide/tutorials/general-NARX-models/" class="feature-link">
			<h3>🛠️ Estimadores Gerais</h3>
		</a>
		<p>Compatível com <strong>scikit-learn</strong>, <strong>CatBoost</strong> e mais para criar modelos NARMAX.</p>
	</div>
</div>

## Recursos adicionais

<ul class="custom-link-list">
	<li>
		<a href="https://sysidentpy.org/developer-guide/contribute/" target="_blank">🤝 Contribua com o SysIdentPy</a>
	</li>
	<li>
		<a href="https://sysidentpy.org/getting-started/license/" target="_blank">📜 Informações de Licença</a>
	</li>
	<li>
		<a href="https://sysidentpy.org/community-support/get-help/" target="_blank">🆘 Ajuda & Suporte</a>
	</li>
	<li>
		<a href="https://sysidentpy.org/community-support/meetups/ai-networks-meetup/" target="_blank">📅 Palestras</a>
	</li>
	<li>
		<a href="https://sysidentpy.org/landing-page/sponsor/" target="_blank">💖 Torne-se um Patrocinador</a>
	</li>
	<li>
		<a href="https://sysidentpy.org/user-guide/API/narmax-base/" target="_blank">🧩 Explore o Código Fonte</a>
	</li>
</ul>

## Você gosta do **SysIdentPy**?

Gostaria de ajudar o SysIdentPy, outros usuários e o criador da biblioteca? Você pode "dar uma estrela" ao projeto no GitHub clicando no botão de estrela no canto superior direito da página: <a href="https://github.com/wilsonrljr/sysidentpy" class="external-link" target="_blank">https://github.com/wilsonrljr/sysidentpy</a>. ⭐️

Ao marcar um repositório com estrela, você o encontra mais facilmente no futuro, recebe sugestões de projetos relacionados no GitHub e ainda valoriza o trabalho do mantenedor.

Considere, também, apoiar o projeto tornando-se um sponsor. Seu apoio ajuda a manter o desenvolvimento ativo e garante a evolução contínua do <strong>SysIdentPy</strong>.

[:octicons-star-fill-24:{ .mdx-heart } &nbsp; Seja um <span class="mdx-sponsorship-count" data-mdx-component="sponsorship-count"></span> Patrocinador no GitHub][wilsonrljr's sponsor profile]{ .md-button .md-button--primary .mdx-sponsorship-button }

  [wilsonrljr's sponsor profile]: https://github.com/sponsors/wilsonrljr
