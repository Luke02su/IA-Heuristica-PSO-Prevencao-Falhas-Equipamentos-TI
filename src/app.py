import tkinter as tk
from ui.interface import App

if __name__ == "__main__":
    # É essencial que o script seja executado a partir do diretório raiz
    try:
        root = tk.Tk()
        root.geometry("450x300")
        App(root)
        root.mainloop()
    except Exception as e:
        print("\n--- ERRO CRÍTICO NA INICIALIZAÇÃO ---")
        print("Verifique se você está executando 'python app.py' a partir do diretório raiz do projeto.")
        print(f"Detalhe do Erro: {e}")
        print("--- FIM DO ERRO ---")