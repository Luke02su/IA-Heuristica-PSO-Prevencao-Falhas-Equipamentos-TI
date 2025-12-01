import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# --- CLASSE PSO ---
class PSO:
    """
    Implementa o algoritmo Particle Swarm Optimization (PSO).
    Otimiza uma função (func) dentro de um espaço dimensional (dim) e limites (bounds).
    """
    def __init__(self, func, dim, bounds, num_particles=15, max_iter=30, w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.dim = dim
        self.bounds = bounds  # [min_vals, max_vals]
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self):
        """Executa o processo de otimização e retorna a melhor solução (gbest)."""
        
        # 1. Inicializa posições (particles) e velocidades (velocities)
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        
        # 2. Inicializa pbest e gbest
        pbest = particles.copy()
        pbest_scores = np.array([self.func(p) for p in particles])
        gbest = pbest[np.argmin(pbest_scores)].copy()

        # 3. Loop principal de otimização
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                
                # Atualiza a velocidade
                # v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - particles[i])
                    + self.c2 * r2 * (gbest - particles[i])
                )
                
                # Atualiza a posição
                particles[i] += velocities[i]
                
                # Aplica o limite (clamping)
                particles[i] = np.clip(particles[i], self.bounds[0], self.bounds[1])

                # Avalia o score da nova posição
                score = self.func(particles[i])
                
                # Atualiza pbest
                if score < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = score

            # Atualiza gbest
            gbest = pbest[np.argmin(pbest_scores)].copy()

        return gbest

# --- CLASSE FailurePredictor ---
class FailurePredictor:
    """
    Classe para treinar um modelo de Machine Learning (Random Forest)
    com otimização de hiperparâmetros usando PSO.
    """
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X, y):
        """
        Treina o modelo, otimizando os hiperparâmetros (n_estimators e max_depth)
        usando o algoritmo PSO.
        """
        
        # Define a função de custo para o PSO: minimizar o erro do modelo
        def cost(params):
            """Calcula a pontuação de custo (erro) para um dado conjunto de parâmetros."""
            try:
                # Converte para int e garante valor mínimo de 1
                n_estimators = max(1, int(params[0]))
                depth = max(1, int(params[1]))
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=depth,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Usa Negative Mean Squared Error (PSO minimiza o erro positivo)
                scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                return -np.mean(scores)
            except Exception as e:
                # print(f"Erro na avaliação do custo: {e}. Retornando custo alto.")
                return 99999.0

        # Configura os limites para os hiperparâmetros: [n_estimators (1 a 200), max_depth (1 a 50)]
        bounds = [
            (1.0, 200.0), # n_estimators
            (1.0, 50.0)   # max_depth
        ]
        
        # Ajusta para o formato esperado pelo PSO: [min_array, max_array]
        pso_bounds = [np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])]

        # Inicializa e executa o PSO
        pso = PSO(func=cost, dim=2, bounds=pso_bounds, max_iter=30, num_particles=15)
        best_params = pso.optimize()

        # Extrai os melhores parâmetros otimizados (convertendo para inteiro)
        best_n_estimators = max(1, int(best_params[0]))
        best_max_depth = max(1, int(best_params[1]))
        
        print(f"PSO Otimizou Parâmetros: n_estimators={best_n_estimators}, max_depth={best_max_depth}")

        # Treina o modelo final com os hiperparâmetros otimizados
        self.model = RandomForestClassifier(
            n_estimators=best_n_estimators,
            max_depth=best_max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

    def predict(self, X):
        """Faz a previsão usando o modelo treinado (retorna array de 0s e 1s)."""
        return self.model.predict(X)