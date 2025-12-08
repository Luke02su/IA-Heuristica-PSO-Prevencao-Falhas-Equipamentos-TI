import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler 

class DataLoader:
    #Carrega, pré-processa e aplica Engenharia de Features de Frequência e Temporal.
    # A variável 'Falha' é definida por palavras-chave no Motivo.

    def __init__(self, path):
        self.path = path
        self.target_coluna = 'Falha'

        self.falha_keywords = [
            'substituição', 'substituir', 'troca', 'trocar', 'queimou',
            'devolução', 'devolver', 'retornar', 'danificado', 'problema',
            'defeito',
        ]

        self.colunas_para_remover = [
            'ID Envio', 'Placa', 'Usuário Envio', 'Motivo'
        ]

        self.colunas_categoricas = [
            'Tipo', 'Modelo', 'Origem', 'Loja Destino (ID)'
        ]

    def _load_data(self):
        #Carrega o CSV com correção de compatibilidade e tratamento de erros de linha.
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {self.path}")

        try:
            # Tenta sintaxe moderna
            df = pd.read_csv(self.path, encoding='utf-8', on_bad_lines='skip')
        except (TypeError, UnicodeDecodeError):
            try:
                # Tenta sintaxe antiga
                df = pd.read_csv(self.path, encoding='utf-8', error_bad_lines=False)
            except Exception:
                # Tenta sintaxe antiga com outra codificação
                df = pd.read_csv(self.path, encoding='latin1', error_bad_lines=False)

        if df.empty:
            raise ValueError("O arquivo CSV está vazio ou todas as linhas foram puladas devido a erros.")

        return df

    def _create_target_and_preprocess(self, df):
        #Cria as features de Falha, Frequência, Intervalo Temporal e pré-processa as colunas.

        # 1. CRIAÇÃO DA VARIÁVEL ALVO (Y) - Coluna 'Falha' (via Keywords)
        keywords_regex = '|'.join(self.falha_keywords)
        df['Falha'] = df['Motivo'].astype(str).str.lower().str.contains(
            keywords_regex, na=False
        ).astype(int)

        # 2. CONVERSÃO DA DATA
        if 'Data Envio' in df.columns:
            df['Data Envio'] = pd.to_datetime(df['Data Envio'], errors='coerce')
            df.dropna(subset=['Data Envio'], inplace=True) # Remove linhas com data inválida (poucas)

        # 3. ENGENHARIA DA FEATURE DE FREQUÊNCIA DE ENVIO (Reenvio)
        if 'Nº Série Equip.' in df.columns:
            contagem_envios = df.groupby('Nº Série Equip.').size().reset_index(name='Frequencia_Envio')
            df = pd.merge(df, contagem_envios, on='Nº Série Equip.', how='left')

        # 4. ENGENHARIA DA FEATURE DE INTERVALO DE DIAS ENTRE REENVIOS ()
        if 'Data Envio' in df.columns and 'Nº Série Equip.' in df.columns:
            # Ordena para garantir que o cálculo do delta seja sequencial
            df = df.sort_values(by=['Nº Série Equip.', 'Data Envio'])

            # Calcula a data de envio ANTERIOR
            df['Data_Envio_Anterior'] = df.groupby('Nº Série Equip.')['Data Envio'].shift(1)

            # Calcula a diferença em dias
            df['Intervalo_Dias_Reenvio'] = (df['Data Envio'] - df['Data_Envio_Anterior']).dt.days

            # Remoção temporária de colunas de data/referência para o próximo passo
            df = df.drop(columns=['Data_Envio_Anterior'])

            # Features de Data baseadas no envio atual
            df['Dia_da_Semana'] = df['Data Envio'].dt.dayofweek.fillna(0).astype(int)
            df['Mes'] = df['Data Envio'].dt.month.fillna(0).astype(int)
            
            # CRIAÇÃO DO FATOR DE RISCO CRÔNICO AGREGADO
            # Combina Frequência alta com Intervalo baixo (falha recorrente e rápida)
            df['Risco_Cronico_Agregado'] = (df['Frequencia_Envio'] / (df['Intervalo_Dias_Reenvio'].fillna(0) + 1))
            
            df = df.drop(columns=['Data Envio'])
        else:
             # Se não houver data, cria as colunas como 0 para manter a consistência do modelo
             df['Frequencia_Envio'] = 0
             df['Intervalo_Dias_Reenvio'] = 0 
             df['Dia_da_Semana'] = 0
             df['Mes'] = 0
             df['Risco_Cronico_Agregado'] = 0


        # 5. LIMPEZA E REMOÇÃO DE COLUNAS DE ID
        colunas_a_remover = self.colunas_para_remover + ['Nº Série Equip.'] 
        colunas_para_remover_temp = [col for col in colunas_a_remover if col in df.columns]

        df_processado = df.drop(
            columns=colunas_para_remover_temp,
            errors='ignore'
        )

        # 6. ONE-HOT ENCODING (Categoricas)
        cols_ohe = [col for col in self.colunas_categoricas if col in df_processado.columns]
        df_final = pd.get_dummies(df_processado, columns=cols_ohe, dummy_na=False)

        # 7. GARANTE QUE TODOS OS DADOS SÃO NUMÉRICOS E TRATA RESTANTES VALORES NULOS
        df_final.fillna(0, inplace=True)

        return df_final


    def load(self):
        """Carrega, pré-processa, aplica escalonamento e retorna X, y, feature_names e colunas de contexto."""
        df_raw = self._load_data()
        self.df_processed = self._create_target_and_preprocess(df_raw.copy())

        if self.target_coluna not in self.df_processed.columns:
            raise ValueError("Erro de pré-processamento: A coluna 'Falha' (target) não foi criada.")

        # --- CAPTURANDO COLUNAS DE CONTEXTO ---
        context_df = df_raw.iloc[self.df_processed.index].copy()

        if 'Nº Série Equip.' in context_df.columns and 'Motivo' in context_df.columns:
             # Manter a coluna Nº Série Equip. no contexto raw para o output
             colunas_contexto = context_df[['Nº Série Equip.', 'Motivo']].values
        else:
             raise ValueError("Colunas 'Nº Série Equip.' ou 'Motivo' não encontradas no arquivo raw.")
        # ----------------------------------------

        y = self.df_processed[self.target_coluna].values

        # Filtra apenas colunas numéricas (int, float) para garantir que não haja strings
        df_features = self.df_processed.drop(columns=[self.target_coluna]).select_dtypes(include=[np.number])

        feature_names = df_features.columns.tolist()
        X = df_features.values

        # MUDANÇA CRÍTICA: Usa StandardScaler para Padronizar (mitigar picos)
        # O StandardScaler é melhor para Random Forest do que o MinMax, pois não esmaga outliers.
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y, feature_names, colunas_contexto