import pandas as pd
import numpy as np
import os

class DataLoader:
    """
    Classe responsável por carregar e pré-processar o arquivo de dados CSV
    'tabelaEnvios.csv' para o modelo de previsão de falhas.
    """
    def __init__(self, path):
        self.path = path
        
        # Palavras-chave para definir se o motivo é uma falha
        self.falha_keywords = [
            'substituição', 'troca', 'queimou', 'danificado', 
            'problema', 'defeito', 'listra', 'trincou', 'não está segurando carga'
        ]

        # Colunas a serem removidas por serem apenas identificadores
        self.colunas_para_remover = [
            'ID Envio', 'Nº Série Equip.', 'Placa', 'Usuário Envio'
        ]

        # Colunas categóricas para aplicar One-Hot Encoding
        self.colunas_categoricas = [
            'Tipo', 'Modelo', 'Motivo', 'Origem', 'Loja Destino (ID)'
        ]

    def _load_data(self):
        """Carrega o CSV tratando possíveis erros de codificação e existência."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {self.path}")

        try:
            df = pd.read_csv(self.path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.path, encoding='latin1')
            
        if df.empty:
            raise ValueError("O arquivo CSV está vazio.")
            
        return df

    def _create_target_and_preprocess(self, df):
        """Cria a variável target 'Falha' e executa o pré-processamento das features."""
        
        # 1. CRIAÇÃO DA VARIÁVEL ALVO (Y) - Coluna 'Falha'
        keywords_regex = '|'.join(self.falha_keywords)
        df['Falha'] = df['Motivo'].astype(str).str.lower().str.contains(
            keywords_regex, na=False
        ).astype(int)
        
        # 2. LIMPEZA INICIAL
        df_processado = df.drop(
            columns=[col for col in self.colunas_para_remover if col in df.columns], 
            errors='ignore'
        )

        # 3. ENGENHARIA DE FEATURES DE DATA
        if 'Data Envio' in df_processado.columns:
            df_processado['Data Envio'] = pd.to_datetime(df_processado['Data Envio'], errors='coerce')
            df_processado['Dia_da_Semana'] = df_processado['Data Envio'].dt.dayofweek.fillna(-1).astype(int)
            df_processado['Mes'] = df_processado['Data Envio'].dt.month.fillna(-1).astype(int)
            df_processado = df_processado.drop(columns=['Data Envio'])

        # 4. ONE-HOT ENCODING
        cols_ohe = [col for col in self.colunas_categoricas if col in df_processado.columns]
        df_final = pd.get_dummies(df_processado, columns=cols_ohe, dummy_na=False)

        # 5. GARANTE QUE TODOS OS DADOS SÃO NUMÉRICOS E TRATA RESTANTES NANs
        df_final.fillna(0, inplace=True)
        
        return df_final

    def load(self):
        """Carrega, pré-processa, e separa features (X) e labels (y) como arrays NumPy."""
        df_raw = self._load_data()
        self.df_processed = self._create_target_and_preprocess(df_raw.copy())
        
        if 'Falha' not in self.df_processed.columns:
            raise ValueError("Erro de pré-processamento: A coluna 'Falha' (target) não foi criada.")

        # Separa o target (y) das features (X)
        y = self.df_processed['Falha'].values
        X = self.df_processed.drop(columns=['Falha']).values

        return X, y