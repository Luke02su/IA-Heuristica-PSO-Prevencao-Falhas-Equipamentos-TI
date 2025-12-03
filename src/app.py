import tkinter as tk
import os
import sys

# Adiciona o diretório PARENT (o diretório do projeto, onde 'ui' está) ao sys.path
# Isso permite que a importação 'from ui.interface import App' funcione.
# A linha abaixo assume que 'src' e 'ui' estão no mesmo nível.
# Ex: projeto_pso/src/app.py -> Adiciona projeto_pso/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa a classe App do módulo interface (localizado em ui/interface.py)
try:
    from ui.interface import App
except ImportError as e:
    # Se a importação falhar mesmo com o path ajustado, levanta o erro para diagnóstico
    print(f"Erro ao tentar importar 'App': {e}")
    sys.exit(1)


if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.title("Predição de Falhas com PSO - TI")
        # Note: O tamanho da janela é definido na classe App, mas você pode definir aqui se preferir.
        
        app_instance = App(root)
        root.mainloop()
    except Exception as e:
        print("\n--- ERRO CRÍTICO NA INICIALIZAÇÃO ---")
        print("Verifique a estrutura do projeto e as dependências instaladas.")
        print(f"Detalhe do Erro: {e}")
        print("--- FIM DO ERRO ---")