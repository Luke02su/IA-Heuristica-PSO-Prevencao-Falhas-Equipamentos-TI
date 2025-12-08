import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import pandas as pd
from operator import itemgetter 

# Importações relativas para a estrutura do projeto
try:
    from src.data_loader import DataLoader
    from src.model import FailurePredictor
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.data_loader import DataLoader
    from src.model import FailurePredictor


class App:

    # Interface Gráfica Tkinter para previsão de falhas (Ranking de Risco)
    def __init__(self, root):
        self.root = root
        self.root.title("Predição de Falhas com PSO - TI")
        self.csv_path = None
        
        window_width = 450
        window_height = 400
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        self.setup_ui()
        self.model = FailurePredictor()
        
        # Variáveis do Data Loader
        self.X = None
        self.y = None
        self.feature_names = None
        self.context_cols = None # NOVO: Array com [Nº Série, Motivo]
        
        # Variáveis do Modelo
        self.predictions = None # Armazenará probabilidades (float)
        self.metrics = None
        self.THRESHOLD = 0.6# Limiar de classificação para Alerta (P > 0.7)

    def setup_ui(self):
        # Cria e organiza os widgets da interface.
        
        main_frame = tk.Frame(self.root, padx=20, pady=10)
        main_frame.pack(expand=True, fill='both')
        
        tk.Label(main_frame, text="Predição de Falhas Otimizada com PSO", font=('Arial', 14, 'bold')).pack(pady=5)
        
        # 1. Carregar CSV
        tk.Label(main_frame, text="1. Selecione o arquivo CSV:").pack(anchor='w', pady=(5, 2))
        self.load_btn = tk.Button(main_frame, text="Abrir CSV", command=self.load_csv, bg="#4CAF50", fg="white", relief=tk.RAISED)
        self.load_btn.pack(fill='x', pady=2)
        self.status_label = tk.Label(main_frame, text="Nenhum arquivo carregado.", fg="red")
        self.status_label.pack(anchor='w')

        # 2. Treinar e Prever
        tk.Label(main_frame, text="2. Otimizar e Treinar o Modelo:").pack(anchor='w', pady=(5, 2))
        self.predict_btn = tk.Button(main_frame, text="Treinar e Prever", command=self.run_model, state=tk.DISABLED, bg="#2196F3", fg="white", relief=tk.RAISED)
        self.predict_btn.pack(fill='x', pady=2)

        # 3. Resultado Principal
        tk.Label(main_frame, text="3. Resumo da Previsão:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(5, 2))
        self.result_label = tk.Label(main_frame, text="Previsão aparecerá aqui.", fg="blue", font=('Arial', 11))
        self.result_label.pack(anchor='w')

        # 4. Métricas de Desempenho
        tk.Label(main_frame, text="4. Métricas (Teste):", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(5, 2))
        self.metrics_label = tk.Label(main_frame, text="Aguardando treinamento...", fg="gray")
        self.metrics_label.pack(anchor='w')
        
        # 5. Botão Exportar
        tk.Label(main_frame, text="5. Ações:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(5, 2))
        self.export_btn = tk.Button(main_frame, text="Exportar Ranking de Risco (CSV)", command=self.export_alerts, state=tk.DISABLED, bg="#FF9800", fg="white", relief=tk.RAISED)
        self.export_btn.pack(fill='x', pady=2)

    def load_csv(self):
        # Abre o diálogo de arquivo e carrega os dados (Recebendo 4 variáveis).
        self.csv_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("Arquivos CSV", "*.csv")]
        )
        if self.csv_path:
            try:
                loader = DataLoader(self.csv_path)
                
                # ATUALIZADO: Recebe 4 variáveis do load()
                self.X, self.y, self.feature_names, self.context_cols = loader.load()
                
                # Carrega o DF original (apenas para referência na exportação)
                try:
                    self.original_df = pd.read_csv(self.csv_path, encoding='utf-8', on_bad_lines='skip')
                except UnicodeDecodeError:
                    self.original_df = pd.read_csv(self.csv_path, encoding='latin1', on_bad_lines='skip')

                if self.X.size == 0 or self.y.size == 0:
                    raise ValueError("Os dados carregados estão vazios ou incompletos.")

                messagebox.showinfo("Sucesso", f"CSV carregado com {self.X.shape[0]} registros!")
                self.status_label.config(text=f"Arquivo: {os.path.basename(self.csv_path)}", fg="green")
                self.predict_btn.config(state=tk.NORMAL)
                self.export_btn.config(state=tk.DISABLED)
                self.metrics_label.config(text="Aguardando treinamento...", fg="gray")
            except Exception as e:
                messagebox.showerror("Erro de Carregamento", str(e))
                self.status_label.config(text="Erro ao carregar o arquivo.", fg="red")
                self.predict_btn.config(state=tk.DISABLED)

    def show_predictions(self, probabilities):
        # Cria uma nova janela e exibe o Ranking de Risco (Probabilidade > 0.7).
        
        # 1. Cria um DataFrame de trabalho a partir das colunas de contexto e probabilidades
        df_ranking = pd.DataFrame(self.context_cols, columns=['Nº Série Equip.', 'Motivo'])
        df_ranking['Probabilidade'] = probabilities
        
        # 2. Filtra apenas os alertas (Probabilidade > THRESHOLD)
        df_alerts = df_ranking[df_ranking['Probabilidade'] >= self.THRESHOLD].copy()
        
        # 3. ORDENAÇÃO: Ranking decrescente pela probabilidade de falha
        df_alerts = df_alerts.sort_values(by='Probabilidade', ascending=False)
        
        falhas_count = len(df_alerts)

        if falhas_count == 0:
            messagebox.showinfo("Resultado da Previsão", "Nenhuma falha foi predita (probabilidade abaixo do limiar de 70%).")
            return

        result_window = tk.Toplevel(self.root)
        result_window.title(f"🏆 RANKING DE RISCO - {falhas_count} Equipamentos Prioritários")
        
        cols_display = ['Probabilidade', 'Nº Série Equip.', 'Motivo']
        
        tree = ttk.Treeview(result_window, columns=cols_display, show='headings')
        tree.pack(expand=True, fill='both')
        
        tree.heading('Probabilidade', text='Risco (%)')
        tree.column('Probabilidade', width=100, anchor=tk.CENTER)
        tree.heading('Nº Série Equip.', text='Nº Série Equip.')
        tree.column('Nº Série Equip.', width=150, anchor=tk.CENTER)
        tree.heading('Motivo', text='Motivo Original')
        tree.column('Motivo', width=300)

        for index, row_data in df_alerts.iterrows():
            probabilidade_percentual = f"{row_data['Probabilidade']:.2%}"
            num_serie = row_data['Nº Série Equip.']
            motivo = row_data['Motivo']
            
            tree.insert('', tk.END, values=(probabilidade_percentual, num_serie, motivo))

    def export_alerts(self):
        # Exporta o Ranking de Risco (Probabilidades > THRESHOLD) para um novo CSV.
        if self.predictions is None or self.context_cols is None:
            messagebox.showwarning("Aviso", "Por favor, treine o modelo primeiro.")
            return

        try:
            # 1. Cria DF de trabalho
            df_work = pd.DataFrame(self.context_cols, columns=['Nº Série Equip.', 'Motivo'])
            df_work['Probabilidade_Risco'] = self.predictions
            
            # 2. Filtra pelo Limiar (THRESHOLD)
            df_alerts = df_work[df_work['Probabilidade_Risco'] >= self.THRESHOLD].copy()
            
            # 3. Ordena para o Ranking
            df_alerts = df_alerts.sort_values(by='Probabilidade_Risco', ascending=False)
            
            if len(df_alerts) == 0:
                messagebox.showinfo("Exportação", "Nenhum equipamento atingiu o limiar de risco para exportação.")
                return

            # Pede ao usuário o local para salvar
            output_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("Arquivos CSV", "*.csv")],
                initialfile="Ranking_Risco_PSO.csv"
            )

            if output_path:
                # Formata a coluna de probabilidade para % antes de salvar
                df_alerts['Probabilidade_Risco'] = df_alerts['Probabilidade_Risco'].apply(lambda x: f'{x:.4f}')
                
                df_alerts.to_csv(output_path, index=False)
                messagebox.showinfo("Sucesso", f"Ranking de Risco exportado com sucesso para:\n{output_path}")

        except Exception as e:
            messagebox.showerror("Erro de Exportação", f"Falha ao exportar ranking: {e}")


    def run_model(self):
        #Executa o treinamento do modelo (com PSO) e faz a previsão de PROBABILIDADE.
        if self.X is None or self.y is None:
            messagebox.showwarning("Aviso", "Por favor, carregue o arquivo CSV primeiro.")
            return

        try:
            self.result_label.config(text="Treinando e Otimizando (Aguarde)...", fg="orange")
            self.metrics_label.config(text="Calculando métricas e importâncias...", fg="orange")
            self.root.update() 

            self.metrics = self.model.train(self.X, self.y)

            # Faz a PREVISÃO EM PROBABILIDADES
            self.predictions = self.model.predict(self.X)
            
            # Classificação binária baseada no THRESHOLD para o resumo
            falhas_binarias = (self.predictions >= self.THRESHOLD).astype(int)
            falhas_count = np.sum(falhas_binarias)

            # --- 1. Exibe Métricas e Importância ---
            # ... (Lógica de exibição das métricas e importâncias) ...
            metrics_display = (
                f"Acurácia: {self.metrics['Accuracy']:.4f} | Precisão: {self.metrics['Precision']:.4f}\n"
                f"Recall: {self.metrics['Recall']:.4f} | F1-Score: {self.metrics['F1-Score']:.4f}"
            )
            self.metrics_label.config(text=metrics_display, fg="black")
            
            if self.feature_names is not None and self.model.feature_importances_ is not None:
                importances = self.model.feature_importances_
                
                feature_importance_list = list(zip(self.feature_names, importances))
                feature_importance_list.sort(key=itemgetter(1), reverse=True)
                
                top_features_text = "Top 5 Fatores de Risco:\n" + "\n".join(
                    [f"- {name}: {importance:.3f}" for name, importance in feature_importance_list[:5]]
                )
            
                messagebox.showinfo("Resultados Detalhados", 
                                    f"Parâmetros Otimizados (PSO):\n"
                                    f"N_Estimators: {self.model.best_params['n_estimators']}, Max_Depth: {self.model.best_params['max_depth']}\n\n"
                                    f"{top_features_text}")
            
            # --- 2. Atualiza Resultado Principal e Ações ---
            result_text = f"Otimização Concluída. {falhas_count} alertas de risco (Prob. >= {self.THRESHOLD:.0%}) preditos."
            self.result_label.config(text=result_text, fg="blue")
            
            self.show_predictions(self.predictions)
            self.export_btn.config(state=tk.NORMAL)
                
        except Exception as e:
            messagebox.showerror("Erro de Processamento", f"Falha ao treinar ou prever: {e}")
            self.result_label.config(text="Erro de processamento.", fg="red")
            self.metrics_label.config(text="Falha ao calcular.", fg="red")
            self.export_btn.config(state=tk.DISABLED)