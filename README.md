# 📝 Sistema de Predição de Falhas Otimizada (Manutenção Preditiva)

## 1\. Definição do Problema e Escolha da Técnica IAC

**Problema Escolhido:** Otimização da Manutenção Preditiva para Previsão de **Reenvio de Equipamentos** (Falha Crônica e Imediata).

**Contexto e Relevância:** O sistema transforma dados históricos de envios em **Alertas de Risco**, permitindo que o gestor priorize a substituição (em vez do reparo) dos itens com maior chance de falhar novamente, reduzindo custos operacionais e o tempo de inatividade.

**Técnica IAC Selecionada:** **Inteligência Coletiva (Particle Swarm Optimization - PSO)**.

  * **Justificativa da IAC:** O PSO é aplicado para *otimização global* dos hiperparâmetros (Número de Estimadores e Profundidade Máxima) do modelo Random Forest. Sua aplicação garante que os parâmetros sejam ajustados para **maximizar o Recall** (minimizando Falsos Negativos), elevando o rigor técnico da solução em um problema de classificação desbalanceada.

-----

## 2\. Instruções de Execução e Dependências (Reprodutibilidade)

### 2.1. Requisitos de Ambiente

Este projeto requer Python 3.8+ e as seguintes bibliotecas. Utilize um ambiente virtual (`virtualenv` ou `conda`) para garantir a reprodutibilidade.

O arquivo de dependências (`requirements.txt`) deve conter:

```text
numpy
pandas
scikit-learn
imbalanced-learn # Para o balanceamento de classes (SMOTE)
tkinter          # Para a interface gráfica de usuário (GUI)
```

### 2.2. Instalação de Dependências

Execute no terminal:

```bash
pip install -r requirements.txt
```

### 2.3. Execução da Aplicação

1.  Certifique-se de que o arquivo de dados (`tabelaEnvios.csv`) esteja acessível no diretório.
2.  Execute o arquivo principal no terminal:

<!-- end list -->

```bash
py src/app.py
```

3.  Na interface gráfica (GUI), clique em **"Abrir CSV"** e selecione o arquivo de dados.
4.  Clique em **"Treinar e Prever"**. O sistema iniciará a otimização por PSO e, em seguida, gerará o ranqueamento de risco.

-----

## 3\. Detalhamento da Implementação da IAC (Critério: Código e Originalidade)

### A. Otimização por PSO (`src/model.py`)

O PSO otimiza `n_estimators` e `max_depth` do Random Forest.

  * **Função de Custo (Fitness):** É definida como **`1 - Recall`**. O PSO minimiza essa função, o que equivale a **maximizar o Recall** na Validação Cruzada (CV), direcionando o modelo para a máxima detecção de falhas.
  * **Balanceamento Integrado:** O modelo utiliza **SMOTE** no treino para criar amostras sintéticas e aplica **`class_weight='balanced'`**, aumentando a robustez da previsão da classe minoritária.

### B. Feature Engineering e Uso de Datas (`src/data_loader.py`)

O **`DataLoader`** é a peça central que processa as datas e envios, transformando-as em indicadores de risco:

1.  **Cálculo da Frequência de Envio (Quantidade):**

      * A coluna `Data Envio` é agrupada pelo `Nº Série Equip.` e é utilizada para calcular o **`Frequencia_Envio`**. Esta *feature* mede quantas vezes o equipamento foi enviado, sendo um indicador de **problema crônico e recorrente**.

2.  **Cálculo do Intervalo de Dias de Reenvio (Tempo):**

      * O **`Intervalo_Dias_Reenvio`** é calculado a partir da diferença entre a `Data Envio` atual e a `Data Envio` anterior para cada equipamento.
      * Um valor **baixo** (curto intervalo de tempo) indica que a falha é *imediata* após o último reparo, sendo um forte sinal de risco.

3.  **Escalonamento:** Todas as *features* numéricas, incluindo as métricas de tempo e frequência, são normalizadas via **`MinMaxScaler`** para garantir que a Otimização por PSO as considere de forma balanceada.

-----

## 4\. Usabilidade e Robustez da Solução

  * **Interface e Usabilidade:** Desenvolvida em **Tkinter**, com fluxo sequencial e mensagens em Português, garantindo a usabilidade para o usuário leigo.
  * **Robustez:** O `DataLoader` inclui tratamento para erros de conversão de data, *strings* remanescentes e valores `NaN` (`fillna(0)`), evitando *crashes* previsíveis.
  * **Saída Prática:** O resultado é o **Ranking de Risco (CSV)** exportável, que transforma a previsão de probabilidade em uma **ferramenta acionável** para a gestão de manutenção.
