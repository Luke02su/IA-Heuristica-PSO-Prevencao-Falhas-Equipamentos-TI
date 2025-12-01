import random
import numpy as np

class PSO:
    """
    Implementa o algoritmo Particle Swarm Optimization (PSO).

    Otimiza uma função (func) dentro de um espaço dimensional (dim) e limites (bounds),
    retornando o melhor vetor de parâmetros encontrado.
    """
    def __init__(self, func, dim, bounds, num_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5):
        """
        Inicializa o algoritmo PSO.

        Args:
            func (function): A função de custo a ser minimizada.
            dim (int): O número de dimensões do problema (número de parâmetros a otimizar).
            bounds (list): Uma lista [min_val, max_val] para os limites de busca.
            num_particles (int): Número de partículas no enxame.
            max_iter (int): Número máximo de iterações.
            w (float): Peso de inércia.
            c1 (float): Coeficiente de aceleração cognitiva (pbest).
            c2 (float): Coeficiente de aceleração social (gbest).
        """
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self):
        """Executa o processo de otimização e retorna a melhor solução."""
        # 1. Inicializa as posições e velocidades das partículas
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        
        # 2. Inicializa pbest (melhor posição individual) e gbest (melhor posição global)
        pbest = particles.copy()
        pbest_scores = np.array([self.func(p) for p in particles])
        gbest = pbest[np.argmin(pbest_scores)]

        # 3. Loop principal de otimização
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                
                # Atualiza a velocidade
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - particles[i])  # Componente Cognitiva (Pbest)
                    + self.c2 * r2 * (gbest - particles[i])     # Componente Social (Gbest)
                )
                
                # Atualiza a posição
                particles[i] += velocities[i]
                
                # Aplica o limite (clamping) nas posições
                particles[i] = np.clip(particles[i], self.bounds[0], self.bounds[1])

                # Avalia o score da nova posição
                score = self.func(particles[i])
                
                # Atualiza pbest
                if score < pbest_scores[i]:
                    pbest[i] = particles[i]
                    pbest_scores[i] = score

            # Atualiza gbest
            gbest = pbest[np.argmin(pbest_scores)]

        return gbest