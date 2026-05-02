# Predição de Umidade do Solo em Diferentes Profundidades Através de Modelos de Séries Temporais

**Autor:** Antonio Carlos da Silva  
**Orientador:** Prof. Dr. Felipe Leite Coelho da Silva  
**Instituição:** Universidade Federal Rural do Rio de Janeiro (UFRRJ) — Instituto de Ciências Exatas, Departamento de Matemática  
**Curso:** Pós-Graduação Latu Sensu em Estatística Aplicada  
**Ano:** 2025  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Antonioufrrj/especializa-o-estatistica/blob/main/trabalho_final_especializa%C3%A7%C3%A3o.ipynb)

---

## Visão Geral do Projeto

### Contexto e Problema

<cite index="1-15,1-16">A umidade do solo é uma variável de estado crucial no sistema terrestre, controlando a troca de água e energia entre a superfície e a atmosfera. A compreensão de sua dinâmica e predição são essenciais para diversas áreas, como gerenciamento de irrigação, monitoramento de secas e prevenção de desastres naturais, como movimentos de massa e inundações.</cite>

<cite index="1-2237,1-2238">Tradicionalmente, a predição do comportamento da umidade do solo é amplamente baseada em modelos físicos, como HYDRUS, baseados na equação de Richards. Embora os modelos físicos sejam reconhecidos por sua capacidade de explicação, eles apresentam desempenho insatisfatório em aplicações práticas, pois a equação de Richards é altamente não linear, portanto, sua solução numérica frequentemente requer alto poder computacional, além de apresentar forte descontinuidade e instabilidade.</cite>

<cite index="1-2240,1-2241">Com o avanço das tecnologias de sensoriamento, um número crescente de sensores tem sido implantado, resultando na geração contínua de grandes volumes de dados. Nesse contexto, os modelos baseados em dados têm ganhado destaque.</cite>

### Objetivo Principal

<cite index="1-2244,1-2245,1-2246,1-2247">O presente trabalho tem como objetivo realizar um estudo comparativo entre duas abordagens de modelos de previsão de séries temporais (estatística e redes neurais artificiais) aplicadas à umidade do solo em diferentes profundidades da estação Jardim Frei Oreste, localizada no município de Campos do Jordão (SP). Na abordagem estatística, foram considerados o modelo autorregressivo integrado de médias móveis sazonais com variáveis exógenas (SARIMAX). Para a abordagem baseada em redes neurais artificiais, consideramos o modelo de autorregressão de redes neurais (NNAR) e de memória de longo e curto prazo (LSTM).</cite>

<cite index="1-2248,1-2249">Vale notar que existem poucos estudos comparando modelos estatísticos com os de rede neurais para umidade do solo e o uso dessas classes de modelos dentro do mesmo estudo comparativo não foi encontrado na literatura. Portanto, o desenvolvimento deste trabalho pode contribuir para a literatura disponível nesta área.</cite>

---

## Metodologia

### Dados Utilizados

<cite index="1-2355">Os dados de precipitação e umidade do solo utilizados neste estudo foram obtidos das estações localizadas no estado de São Paulo, disponibilizadas pelo Centro Nacional de Monitoramento e Alertas de Desastres Naturais (Cemaden), e estão disponíveis publicamente em seu website.</cite>

As estações são equipadas com o **sensor EnviroSCAN**, que <cite index="1-2379,1-2380">realiza medições de umidade do solo em seis profundidades distintas, variando de 0,5 m a 3,0 m, com intervalos regulares de 0,5 m entre os sensores. A técnica de medição adotada é baseada na capacitância do solo, ou seja, a leitura de umidade do solo é realizada de maneira indireta, e os valores obtidos são posteriormente convertidos em umidade volumétrica por meio</cite> de curvas de calibração.

A estação selecionada para o estudo foi a **Jardim Frei Oreste**, localizada em Campos do Jordão - SP, com dados de **janeiro de 2020 a agosto de 2025**, cobrindo três profundidades principais:

| Profundidade | Média (%) | Desvio Padrão |
|---|---|---|
| 0 – 50 cm | 22,72 | 5,19 |
| 50 – 100 cm | 28,06 | 4,02 |
| 100 – 150 cm | 28,33 | 2,18 |

### Pré-processamento

<cite index="1-2356">O conjunto de dados foi transformado para resolução diária por meio da média da umidade do solo e, posteriormente, dividido em dois subconjuntos: treinamento (de janeiro de 2020 a julho de 2024) e teste (de agosto de 2024 a agosto de 2025), correspondendo a 20% dos dados totais.</cite>

<cite index="1-2357,1-2358">Os dados foram padronizados para evitar viés em relação a qualquer recurso específico devido à sua magnitude. Foi utilizado o método StandardScaler da biblioteca scikit-learn, que transforma os dados para terem média zero e desvio padrão um.</cite>

**Tratamento de outliers:** <cite index="1-2389,1-2390,1-2391">Os valores de umidade do solo que aparecem isolados no gráfico foram considerados outliers, pois podem ter ocorrido devido a interferências no sensor. Esses valores não condizem com a realidade, uma vez que o processo de secagem do solo normalmente não apresenta mudanças abrupta em um intervalo de uma hora entre as leituras. Portanto, esses valores foram excluídos da análise.</cite>

**Variável exógena:** <cite index="1-2431,1-2432,1-2433,1-2434">Propomos o uso da precipitação acumulada (PA) como variável independente para a predição da umidade do solo. O uso dessa variável é amplamente difundido em estudos sobre sistemas de alerta antecipado para movimento de massa. Esses estudos indicam que eventos de precipitação de curta duração exercem pouca influência sobre a umidade do solo, uma vez que a infiltração requer certo tempo e grande parte da água escoa superficialmente. Foram testadas correlações entre a umidade do solo e a precipitação acumulada em janelas de 3, 7 e 14 dias, sendo esta última a que apresentou a melhor correlação.</cite>

### Justificativas das Escolhas Metodológicas

- **Resolução diária:** Reduz ruído e torna a série mais estável para modelagem.
- **Precipitação acumulada em 14 dias:** Melhor correlação com a umidade do solo em comparação com janelas menores.
- **Padronização apenas no treino:** Evita vazamento de informação do conjunto de teste.
- **Seleção de parâmetros por AIC:** Segue o princípio da parcimônia para os modelos SARIMA/SARIMAX.

---

## Modelos Utilizados

### 1. SARIMA / SARIMAX

<cite index="1-2264">O modelo autorregressivo integrado de médias móveis sazonais com regressores externos ou exógenos (SARIMAX) compõe a abordagem estatística adotada.</cite>

<cite index="1-2272,1-2273">Quando as séries temporais apresentam características sazonais utiliza-se o modelo SARIMA(p,d,q)(P,D,Q)S que considera todos os termos do modelo ARIMA(p,d,q) e adiciona os operadores autorregressivo integrado de médias móveis na parte sazonal. O modelo SARIMAX possibilita a utilização de variáveis externas ou exógenas.</cite>

**Modelos selecionados (sem variável exógena):**

| Profundidade | Modelo |
|---|---|
| 0 – 50 cm | SARIMA(0,1,2)(0,1,0)₃₆₅ |
| 50 – 100 cm | SARIMA(2,1,0)(0,1,0)₃₆₅ |
| 100 – 150 cm | SARIMA(2,1,0)(0,1,0)₃₆₅ |

**Modelos selecionados (com variável exógena — precipitação acumulada 14 dias):**

| Profundidade | Modelo |
|---|---|
| 0 – 50 cm | SARIMAX(1,1,0)(0,1,0)₃₆₅ |
| 50 – 100 cm | SARIMAX(2,1,0)(0,1,0)₃₆₅ |
| 100 – 150 cm | SARIMAX(2,1,0)(0,1,0)₃₆₅ |

**Seleção de parâmetros:** <cite index="1-2451,1-2452">Foi aplicado um algoritmo de seleção de parâmetros para o modelo SARIMA, considerando valores entre 0 e 3 para cada parâmetro (p, d, q, P, D, Q). Seguindo o princípio da parcimônia, escolheu-se a estrutura com o menor valor do Critério de Informação de Akaike (AIC).</cite>

---

### 2. NNAR (Neural Network Autoregression)

<cite index="1-2310">O modelo Rede Neural Autorregressiva (NNAR) funciona de forma semelhante à rede MLP, porém, além de considerar os valores atuais das entradas da rede, também considera os valores anteriores, através da inclusão de uma linha de atraso entre as entradas e a rede.</cite>

**Hiperparâmetros utilizados:**

| Parâmetro | Valor (sem exógena) | Valor (com exógena) |
|---|---|---|
| Lags autorregressivos (p) | 5 | 5 |
| Lags sazonais (P) | 1 | 1 |
| Neurônios na camada oculta | 100 | 4 |
| Período sazonal (m) | 365 | 365 |
| Repetições (ensemble) | 20 | 20 |

A biblioteca utilizada foi `arnet` com `MLPRegressor` do scikit-learn como modelo base.

---

### 3. LSTM (Long Short-Term Memory)

<cite index="1-2315,1-2316">A rede de memória de curto e longo prazo (LSTM) foi desenvolvida, em 1997, por Hochreiter e Schmidhuber para superar as limitações das unidades recorrentes tradicionais, que muitas vezes têm dificuldade em lidar com gradiente durante o treinamento em sequências temporais longas. Ao invés de neurônios, as redes LSTM possuem blocos de memória "célula de estado" conectados através de camadas, que atua como uma espécie de memória de longo prazo.</cite>

A LSTM utiliza três portões principais para controlar o fluxo de informações:
- **Forget Gate:** decide o que esquecer da célula de estado
- **Input Gate:** controla o que adicionar à célula de estado
- **Output Gate:** determina o que expor como saída

**Hiperparâmetros utilizados:**

| Parâmetro | Valor |
|---|---|
| Janela temporal (input_chunk_length) | 30 |
| Unidades na camada oculta (hidden_dim) | 20 |
| Batch size | 16 |
| Épocas máximas | 100 |
| Taxa de aprendizado | 0,0005 |
| Otimizador | Adam |

A biblioteca utilizada foi `Darts` com o modelo `BlockRNNModel`.

---

## Resultados

### Métricas de Avaliação

Foram utilizadas quatro métricas de desempenho:

- **MSE** — Erro Quadrático Médio
- **RMSE** — Raiz do Erro Quadrático Médio
- **MAE** — Erro Absoluto Médio
- **MAPE** — Erro Percentual Absoluto Médio

### Comparação sem Variável Exógena

| Profundidade | Métrica | SARIMA | NNAR | LSTM |
|---|---|---|---|---|
| 0 – 50 cm | MSE | 33,49 | **11,05** | 64,03 |
| 0 – 50 cm | MAPE (%) | 20,80 | **13,06** | 33,78 |
| 50 – 100 cm | MSE | **8,15** | 35,62 | 50,05 |
| 50 – 100 cm | MAPE (%) | **7,78** | 17,47 | 23,71 |
| 100 – 150 cm | MSE | **2,67** | 5,55 | 7,04 |
| 100 – 150 cm | MAPE (%) | **4,43** | 6,50 | 8,08 |

### Comparação com Variável Exógena (Precipitação Acumulada 14 dias)

| Profundidade | Métrica | SARIMAX | NNAR | LSTM |
|---|---|---|---|---|
| 0 – 50 cm | MSE | 37,86 | **5,01** | 21,21 |
| 0 – 50 cm | MAPE (%) | 22,83 | **8,21** | 17,65 |
| 50 – 100 cm | MSE | 8,48 | **1,99** | 10,50 |
| 50 – 100 cm | MAPE (%) | 8,16 | **4,18** | 10,68 |
| 100 – 150 cm | MSE | 2,70 | 3,14 | **2,50** |
| 100 – 150 cm | MAPE (%) | 4,73 | 7,79 | **4,41** |

### Interpretação dos Resultados

<cite index="1-2475,1-2476">Nas profundidades de 50–100 cm e 100–150 cm, o modelo SARIMA obteve o melhor desempenho, superando os demais modelos em todas as métricas que foram analisadas. Contudo, ele tende a subestimar os valores de umidade do solo.</cite>

<cite index="1-2478,1-2479">A melhora do SARIMA na camada de 100–150 cm está possivelmente associada à menor variabilidade dos dados nesse nível. Por se tratar de uma camada mais profunda, o solo é menos influenciado por fatores externos, exigindo maior volume de precipitação para provocar alterações perceptíveis na umidade.</cite>

<cite index="1-2523,1-2524,1-2525">De forma geral, a inclusão da variável exógena resultou em uma melhora no desempenho dos modelos de rede neurais. Entre eles, o modelo NNAR apresentou o melhor ajuste aos dados nas camadas mais superficiais (0–50 cm e 50–100 cm), superando os demais modelos em todas as métricas avaliadas. Essa melhoria decorre da capacidade das redes neurais de capturar relações não lineares entre as variáveis, diferentemente do modelo SARIMAX, que é restrito a relações lineares.</cite>

<cite index="1-2482,1-2483,1-2484">O desempenho ruim do modelo LSTM pode estar associado a dois possíveis fatores: a dificuldade do modelo em capturar dependências temporais de longo prazo e/ou o pequeno conjunto de dados utilizados para o processo de treinamento. Diferentemente dos modelos estatísticos tradicionais de séries temporais, no LSTM não se faz mais necessário especificar as quantidades de entradas defasadas que serão necessárias para realizar a previsão. Entretanto, aspectos sazonais de longo período, como o comportamento anual da umidade do solo, que demanda aproximadamente 365 observações, podem ser de difícil aprendizado.</cite>

---

## Conclusões

### Insights Principais

<cite index="1-2541,1-2542">O modelo SARIMA apresentou predições mais consistentes em todas as simulações, demonstrando uma melhora em solo mais profundo onde os dados têm menor variância. Por outro lado, os modelos de rede neural não obtiveram bom desempenho sem o uso da variável exógena, possivelmente pela característica da umidade do solo de ter dependência de longo prazo, e à baixa quantidade de dados disponível no treinamento.</cite>

<cite index="1-2543,1-2544">Com a inclusão da precipitação acumulada como variável exógena, os modelos NNAR e o LSTM apresentaram desempenho bastante satisfatório, pois, de maneira geral, eles conseguem acompanhar a tendência da série temporal. No entanto, ambos ainda demonstraram limitação em capturar pequenas variações diárias com precisão.</cite>

<cite index="1-2545">As predições de umidade do solo obtidas neste estudo podem contribuir para a gestão de desastres naturais, o planejamento da irrigação e a compreensão dos impactos das mudanças climáticas.</cite>

### Limitações do Estudo

- <cite index="1-2487">O conjunto de dados disponível para o treinamento contempla apenas quatro ciclos sazonais completos (4 anos), o que pode limitar a capacidade da rede em generalizar padrões mais complexos e recorrentes na série temporal.</cite>
- <cite index="1-2385">A confiabilidade das medições de umidade do solo não pode ser definitivamente estabelecida, pois não sabemos se foi construída uma curva específica derivada de diferentes tipos de solo, ou a exatidão da correspondência entre as leituras do sensor e medições diretas de umidade do solo no campo.</cite>
- O estudo se limita a uma única estação geotécnica, o que restringe a generalização dos resultados.
- <cite index="1-2546">São necessários estudos adicionais para examinar o desempenho dos modelos de predição de umidade do solo quando utilizados em conjunto com previsões meteorológicas de precipitação geradas por modelos numéricos.</cite>

### Possíveis Melhorias Futuras

<cite index="1-2547">Para pesquisas futuras, pretende-se investigar o efeito de outras variáveis meteorológicas e realizar comparações entre os modelos SARIMAX, NNAR e LSTM com abordagens mais avançadas, como modelos que utilizam mecanismos de auto atenção, capazes de identificar padrões relevantes em diferentes períodos da série temporal, além de compará-los com modelos físicos de umidade do solo, como o HYDRUS.</cite>

Outras melhorias sugeridas:
- Ampliar o estudo para múltiplas estações geotécnicas
- Incorporar variáveis meteorológicas adicionais (temperatura, radiação solar, umidade relativa do ar)
- Explorar modelos híbridos (estatístico + rede neural)
- Aumentar o período de treinamento à medida que novos dados forem coletados

---

## Como Reproduzir

### Pré-requisitos

- Python 3.9+
- Jupyter Notebook ou Google Colab
- Acesso aos dados do [CEMADEN](https://www.cemaden.gov.br/mapainterativo)

### Dependências

Instale as dependências necessárias:

```bash
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install matplotlib==3.10.0
pip install seaborn
pip install plotly
pip install scikit-learn==1.6.1
pip install statsmodels
pip install darts==0.38.0
pip install arnet
pip install tensorflow
pip install geopandas
pip install openpyxl
pip install xarray
pip install requests
pip install beautifulsoup4
```

Ou instale tudo de uma vez:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels "darts[torch]" arnet tensorflow geopandas openpyxl xarray requests beautifulsoup4
```

### Passo a Passo

**1. Clone o repositório**

```bash
git clone https://github.com/Antonioufrrj/especializa-o-estatistica.git
cd especializa-o-estatistica
```

**2. Obtenha os dados**

Acesse o portal do CEMADEN e baixe os dados das estações geotécnicas do estado de São Paulo:
- URL: https://www.cemaden.gov.br/mapainterativo
- Período: janeiro de 2020 a agosto de 2025
- Estação principal: Jardim Frei Oreste (Campos do Jordão - SP)

**3. Organize os dados**

Coloque os arquivos CSV baixados na estrutura de pastas esperada pelo notebook:

```
umidade_solo/
├── todos_municipios/     # CSVs de todas as estações
├── df_frei_orestes.csv   # Dados processados da estação principal
└── df_frei_orestes_d.csv # Dados em resolução diária
```

**4. Execute o notebook**

Abra o notebook no Jupyter ou Google Colab:

```bash
jupyter notebook trabalho_final_especialização.ipynb
```

Ou acesse diretamente no Colab pelo badge no topo deste README.

**5. Fluxo de execução**

Execute as células na seguinte ordem:

1. **Importação de bibliotecas** — carrega todas as dependências
2. **Tratamento das estações** — leitura e limpeza dos dados brutos do CEMADEN
3. **Estações escolhidas** — filtragem e processamento da estação Jardim Frei Oreste
4. **Análise de série temporal** — decomposição, correlação e análise exploratória
5. **Separar dados (train/test)** — divisão 80/20 e padronização
6. **SARIMAX** — ajuste e previsão com o modelo estatístico
7. **NNARX** — ajuste e previsão com a rede neural autorregressiva
8. **LSTM Darts** — ajuste e previsão com a rede LSTM
9. **Métricas** — comparação final dos modelos

### Estrutura do Repositório

```
especializa-o-estatistica/
└── trabalho_final_especialização.ipynb   # Notebook principal com todo o código
```

---

## Referências Principais

- HOCHREITER, S.; SCHMIDHUBER, J. Long short-term memory. *Neural computation*, v. 9, n. 8, p. 1735–1780, 1997.
- HYNDMAN, R. J.; ATHANASOPOULOS, G. *Forecasting: principles and practice*. OTexts, 2018.
- MORETTIN, P. A.; TOLOI, C. *Análise de séries temporais*. 2006.
- VEREECKEN, H. et al. On the spatio-temporal dynamics of soil moisture at the field scale. *Journal of Hydrology*, v. 516, p. 76–96, 2014.
- HERZEN, J. et al. Darts: User-friendly modern machine learning for time series. *Journal of Machine Learning Research*, v. 23, n. 124, p. 1–6, 2022.

---

## Licença

Este projeto está disponível para uso livre, cópia, modificação e distribuição, desde que a fonte seja devidamente citada:

> SILVA, Antonio Carlos da. **Predição de Umidade do Solo em Diferentes Profundidades Através de Modelos de Séries Temporais**. Monografia de Especialização (Estatística Aplicada). Instituto de Ciências Exatas, Universidade Federal Rural do Rio de Janeiro, Seropédica, RJ, 2025.

O uso do código e dos resultados para fins acadêmicos, educacionais ou de pesquisa é encorajado, com a devida atribuição ao autor.
