import tkinter as tk
import os
import sys

# A linha abaixo assume que 'src' e 'ui' estão no mesmo nível.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ui.interface import App
except ImportError as e:
    print(f"Erro ao tentar importar 'App': {e}")
    sys.exit(1)


if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.title("Predição de Falhas com PSO - TI")
        
        app_instance = App(root)
        root.mainloop()
    except Exception as e:
        print("\n--- ERRO CRÍTICO NA INICIALIZAÇÃO ---")
        print("Verifique a estrutura do projeto e as dependências instaladas.")
        print(f"Detalhe do Erro: {e}")
        print("--- FIM DO ERRO ---")