# 📝 Sistema de Predição de Falhas Otimizada (Manutenção Preditiva)

## 1\. Definição do Problema e Escolha da Técnica IAC

### 1.1. Contexto e Objetivo

| Item | Descrição |
| :--- | :--- |
| **Problema Principal** | Otimização da Manutenção Preditiva para Previsão de **Reenvio de Equipamentos** (Falha Crônica ou Imediata). |
| **Relevância** | O sistema converte dados históricos em **Alertas de Risco Acionáveis**, permitindo a gestão priorizar a **substituição preventiva e/ou definitiva** de itens com alta probabilidade de falha recorrente. Isso resulta em redução de custos operacionais e tempo de inatividade (*downtime*). |

-----

### 1.2. Técnica IAC Selecionada: Particle Swarm Optimization (PSO)

| Técnica | Aplicação | Justificativa Técnica |
| :--- | :--- | :--- |
| **Inteligência Coletiva (PSO)** | Otimização global dos hiperparâmetros do modelo (Random Forest): `n_estimators` e `max_depth`. | O PSO garante que os parâmetros sejam ajustados para **maximizar o Recall** (minimizando Falsos Negativos), elevando o rigor da solução em um cenário de classificação desbalanceada de alto custo. |

-----

## 2\. Instruções de Execução e Dependências (Reprodutibilidade)

### 2.1. Requisitos e Dependências

Este projeto requer **Python 3.8+** e as seguintes bibliotecas. Utilize um ambiente virtual (`virtualenv` ou `conda`) para garantir a reprodutibilidade.

O arquivo de dependências (`requirements.txt`) deve conter:

```text
numpy
pandas
scikit-learn
imbalanced-learn
tkinter
scipy
```

Execute a instalação no terminal:

```bash
pip install -r requirements.txt
```

### 2.2. Execução da Aplicação

1.  Certifique-se de que o arquivo de dados (`tabelaEnvios.csv`) esteja acessível no diretório do projeto.
2.  Execute o arquivo principal no terminal:
    ```bash
    py src/app.py
    ```
3.  Na interface gráfica (GUI), clique em **"Abrir CSV"** e selecione o arquivo de dados.
4.  Clique em **"Treinar e Prever"**. O sistema iniciará a otimização por PSO e, em seguida, gerará o ranqueamento de risco.

-----

## 3\. Detalhamento da Implementação da IAC e Modelagem

### A. Otimização por PSO e Treinamento (`src/model.py`)

O PSO otimiza o *Random Forest* com foco na robustez da detecção de falhas.

  * **Função de Custo (Fitness):** Definida como **`1 - Recall`**. O PSO minimiza essa função, resultando na **maximização do Recall** na Validação Cruzada (CV).
  * **Estratégia de Balanceamento:** O modelo utiliza o parâmetro **`class_weight='balanced'`** no Random Forest. Esta abordagem prioriza matematicamente o treinamento na classe minoritária (Falha).
  * **Regularização:** O parâmetro **`min_samples_leaf=5`** impede o *overfitting* ao exigir um número mínimo de amostras por nó folha, criando regras de decisão mais generalizáveis.

### B. Feature Engineering e Processamento de Dados (`src/data_loader.py`)

O `DataLoader` transforma colunas de data e ID em *features* cruciais de risco:

1.  **Cálculo da Frequência de Envio (Quantidade):** Extrai a **`Frequencia_Envio`**, um indicador de problema crônico e recorrente.
2.  **Cálculo do Intervalo de Dias de Reenvio (Tempo):** Calcula o **`Intervalo_Dias_Reenvio`**. Um intervalo **curto** indica falha imediata pós-reparo (alto risco).
3.  **Pré-processamento e Escalonamento:** Todas as *features* numéricas são normalizadas via **`MinMaxScaler`**.

-----

## 4\. Usabilidade, Robustez e Resultados

### 4.1. Saída e Usabilidade

  * **Interface:** Desenvolvida em **Tkinter** com um fluxo sequencial e focado na usabilidade para o gestor.
  * **Saída Prática:** O resultado final é o **Ranking de Risco (CSV) exportável**, que lista os equipamentos por probabilidade decrescente de falha, sendo uma **ferramenta de priorização acionável**.

### 4.2. Robustez e Desempenho

  * **Robustez do `DataLoader`:** Inclui tratamento para `NaN` e conversões de *dtype*, prevenindo erros de carregamento.
  * **Métricas Finais de Desempenho:** A otimização atingiu resultados ideais para Manutenção Preditiva:
      * **Acurácia: $\approx 0.74$** (Boa, tendo em vista que não é a principal métrica).
      * **Recall: $\approx 0.90$** (Detecção de $92\%$ das falhas reais, execelente, sendo principal parâmetro).
      * **Precisão: $\approx 0.77$** (Bom, dada a priorização do Recall).
      * **F1-Score: $\approx 0.84$** (Ótimo equilíbrio geral do modelo).
