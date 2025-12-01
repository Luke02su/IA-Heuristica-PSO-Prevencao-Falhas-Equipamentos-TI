import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import pandas as pd

# Importações relativas para a estrutura do projeto
try:
    # Tenta importação direta (melhor se executado do diretório raiz)
    from src.data_loader import DataLoader
    from src.model import FailurePredictor
except ImportError:
    # Ajuste de caminho (fallback)
    import sys
    # Assume que 'ui' e 'src' estão no mesmo nível acima
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
    from src.data_loader import DataLoader
    from src.model import FailurePredictor


class App:
    """
    Interface Gráfica Tkinter para carregar dados, treinar o modelo PSO-otimizado
    e exibir a previsão de falhas na tabela.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Predição de Falhas com PSO - TI")
        self.csv_path = None
        
        # Configurações iniciais
        window_width = 450
        window_height = 300
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        self.setup_ui()
        self.model = FailurePredictor()
        self.X = None
        self.y = None
        self.original_df = None # Manter o DF original para exibir Nº Série e Motivo

    def setup_ui(self):
        """Cria e organiza os widgets da interface."""
        
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        tk.Label(main_frame, text="Predição de Falhas Otimizada com PSO", font=('Arial', 14, 'bold')).pack(pady=10)
        tk.Label(main_frame, text="1. Selecione o arquivo CSV:").pack(anchor='w', pady=(10, 5))
        
        self.load_btn = tk.Button(main_frame, text="Abrir CSV", command=self.load_csv, bg="#4CAF50", fg="white", relief=tk.RAISED)
        self.load_btn.pack(fill='x', pady=5)
        
        self.status_label = tk.Label(main_frame, text="Nenhum arquivo carregado.", fg="red")
        self.status_label.pack(anchor='w')

        tk.Label(main_frame, text="2. Otimizar e Treinar o Modelo:").pack(anchor='w', pady=(10, 5))
        self.predict_btn = tk.Button(main_frame, text="Treinar e Prever", command=self.run_model, state=tk.DISABLED, bg="#2196F3", fg="white", relief=tk.RAISED)
        self.predict_btn.pack(fill='x', pady=5)

        tk.Label(main_frame, text="3. Resultado:").pack(anchor='w', pady=(10, 5))
        self.result_label = tk.Label(main_frame, text="Previsão aparecerá aqui.", fg="blue", font=('Arial', 12))
        self.result_label.pack(pady=5)

    def load_csv(self):
        """Abre o diálogo de arquivo e carrega os dados."""
        self.csv_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("Arquivos CSV", "*.csv")]
        )
        if self.csv_path:
            try:
                # 1. Carrega e Processa dados para X e Y
                loader = DataLoader(self.csv_path)
                self.X, self.y = loader.load() 
                
                # 2. Carrega o DF original novamente SÓ para ter as colunas identificadoras não processadas (SEM errors='ignore')
                try:
                    self.original_df = pd.read_csv(self.csv_path, encoding='utf-8')
                except UnicodeDecodeError:
                    self.original_df = pd.read_csv(self.csv_path, encoding='latin1')

                if self.X.size == 0 or self.y.size == 0 or self.original_df.empty:
                    raise ValueError("Os dados carregados estão vazios ou incompletos.")

                messagebox.showinfo("Sucesso", f"CSV carregado com {self.X.shape[0]} registros!")
                self.status_label.config(text=f"Arquivo: {os.path.basename(self.csv_path)}", fg="green")
                self.predict_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Erro de Carregamento", str(e))
                self.status_label.config(text="Erro ao carregar o arquivo.", fg="red")
                self.predict_btn.config(state=tk.DISABLED)

    def show_predictions(self, predictions):
        """Cria uma nova janela e exibe as linhas onde a previsão de falha é 1."""
        
        # Filtra apenas as linhas onde a previsão é 1
        falhas_preditas_indices = np.where(predictions == 1)[0]
        
        if len(falhas_preditas_indices) == 0:
            messagebox.showinfo("Resultado da Previsão", "Nenhuma falha foi predita. O modelo não acionou alertas.")
            return

        result_window = tk.Toplevel(self.root)
        result_window.title(f"Alertas de Falha - {len(falhas_preditas_indices)} Equipamentos em Risco")
        
        # Colunas que queremos exibir (dados brutos para identificação)
        cols_display = ['Nº Série Equip.', 'Motivo', 'Previsão']
        
        tree = ttk.Treeview(result_window, columns=cols_display, show='headings')
        tree.pack(expand=True, fill='both')
        
        # Configura as colunas
        tree.heading('Nº Série Equip.', text='Nº Série Equip.')
        tree.column('Nº Série Equip.', width=150, anchor=tk.CENTER)
        tree.heading('Motivo', text='Motivo Original')
        tree.column('Motivo', width=300)
        tree.heading('Previsão', text='Alerta (1)')
        tree.column('Previsão', width=80, anchor=tk.CENTER)

        # Insere os dados filtrados no Treeview
        for index in falhas_preditas_indices:
            # Pega os dados brutos da linha correspondente
            row_data = self.original_df.iloc[index]
            
            num_serie = row_data.get('Nº Série Equip.', 'N/D')
            motivo = row_data.get('Motivo', 'N/D')
            
            tree.insert('', tk.END, values=(num_serie, motivo, 1))

    def run_model(self):
        """Executa o treinamento do modelo (com PSO) e faz a previsão em todos os dados."""
        if self.X is None or self.y is None:
            messagebox.showwarning("Aviso", "Por favor, carregue o arquivo CSV primeiro.")
            return

        try:
            self.result_label.config(text="Treinando e Otimizando (Aguarde)...", fg="orange")
            self.root.update() 

            # Treinamento com Otimização PSO (PSO OTIMIZA o modelo)
            self.model.train(self.X, self.y)

            # Faz a PREVISÃO EM TODOS OS DADOS
            previsoes_totais = self.model.predict(self.X)
            falhas_count = np.sum(previsoes_totais)

            # Atualiza o resultado na janela principal
            result_text = f"Otimização Concluída. {falhas_count} alertas de falha preditos."
            self.result_label.config(text=result_text, fg="blue")
            
            messagebox.showinfo("Treinamento Concluído", f"Modelo Random Forest otimizado com sucesso! {falhas_count} alertas prontos para visualização.")
            
            # EXIBE A TABELA COMPLETA DAS LINHAS COM ALERTA
            self.show_predictions(previsoes_totais)
                
        except Exception as e:
            messagebox.showerror("Erro de Processamento", f"Falha ao treinar ou prever: {e}")
            self.result_label.config(text="Erro de processamento.", fg="red")