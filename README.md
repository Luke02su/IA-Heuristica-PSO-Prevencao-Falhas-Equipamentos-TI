# 🚀 Projeto: Sistema de Manutenção Preditiva Otimizada (PSO-RF)

## 💊 1. Visão Geral do Projeto (Rede Farmácia Nacional)

Este projeto implementa uma solução de **Manutenção Preditiva (PdM)** para a Rede Farmácia Nacional. O objetivo principal é transformar dados históricos de envios de equipamentos de TI em um **Plano de Ação Proativo**, identificando os ativos com maior probabilidade de falha **antes** que o problema cause interrupções nas operações de venda (*downtime*).

A solução gera um **Ranking de Risco** mensal ou semanal para otimizar os recursos da equipe de manutenção.

---

## 🧠 2. Metodologia: Otimização e Previsão

O core do sistema é um modelo **Random Forest (RF)** cujos hiperparâmetros foram ajustados de forma avançada usando o algoritmo **Particle Swarm Optimization (PSO)**.

### A. Otimização com PSO

O algoritmo PSO foi utilizado para encontrar a combinação ideal de **n_estimators** (número de árvores) e **max_depth** (profundidade máxima) do Random Forest.

* **Objetivo:** Minimizar o erro do modelo, maximizando a **capacidade de previsão** das falhas.
* **Métrica de Otimização:** Otimizamos o **F1-Score Ponderado (`f1_weighted`)**. Esta é uma escolha técnica crucial, pois garante que o modelo mantenha um bom equilíbrio entre **Precisão** (reduzindo alarmes falsos) e **Recall** (capturando o máximo de falhas reais) em um cenário de dados desbalanceados.

### B. O Processamento de Dados

A qualidade da previsão é garantida por um pré-processamento robusto:

1.  **Criação do Target (`Falha`):** A variável alvo binária (1/0) é gerada usando uma lista de palavras-chave (`troca`, `defeito`, `queimou`, etc.) encontradas na coluna `Motivo` original.
2.  **Feature Chave: Frequência de Envio:** Foi introduzida a *feature* **`Frequencia_Envio`**, que contabiliza o número de vezes que um equipamento (pelo seu Nº de Série) apareceu no histórico. Esta é uma medida direta da **confiabilidade histórica** do ativo.

---

## 🏆 3. Resultado e Valor para o Negócio

O resultado mais valioso do projeto é a **capacidade de Ranqueamento de Risco**, implementada através da função `predict_proba()`.

### A. Geração do Ranking de Risco

Em vez de uma simples classificação binária (0 ou 1), o modelo retorna a **probabilidade (chance)** do equipamento falhar novamente.

1.  **Previsão de Probabilidade:** O modelo retorna um valor entre 0 e 1 (ex: 0.95 = 95% de chance de falha).
2.  **Ranqueamento:** Os equipamentos que excedem um limite de risco (ex: Probabilidade > 50%) são ordenados de forma decrescente.

### B. Proposta de Integração

O Ranking de Risco deve ser integrado ao sistema de controle de equipamentos (ou exportado mensalmente/semanalmente via CSV) para:

* **Priorização:** A equipe de manutenção prioriza os equipamentos no topo do ranking, garantindo que o tempo e os recursos sejam alocados onde o risco é mais iminente.
* **Decisão de Compra/Descarte:** Fornece dados para justificar a substituição de modelos de equipamentos que consistentemente aparecem no topo da lista.

---

## 🛠️ 4. Configuração e Execução

### Pré-requisitos

Certifique-se de ter o Python 3.x instalado.

### Instalação de Dependências

O projeto requer as seguintes bibliotecas Python, listadas no `requirements.txt`:

```bash
numpy
pandas
scikit-learn
tkinter
