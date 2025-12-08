import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

class PSO:
    #Otimizador por Enxame de Partículas para encontrar os melhores hiperparâmetros.
    def __init__(self, func, dim, bounds, num_particles=10, max_iter=10, w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self):
        # FIXADO: Garante que as posições e velocidades iniciais sejam sempre as mesmas
        np.random.seed(42) 
        
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        pbest = particles.copy()
        pbest_scores = np.array([self.func(p) for p in particles])
        gbest = pbest[np.argmin(pbest_scores)].copy()

        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] + self.c1 * r1 * (pbest[i] - particles[i]) + self.c2 * r2 * (gbest - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.bounds[0], self.bounds[1])
                score = self.func(particles[i])
                if score < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = score
            gbest = pbest[np.argmin(pbest_scores)].copy()
        return gbest

# CLASSE FailurePredictor (COM REGULARIZAÇÃO)
class FailurePredictor:
    #Treina um modelo Random Forest com otimização de hiperparâmetros (PSO) e balanceamento (SMOTE).
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.feature_importances_ = None
        self.best_params = {}

    def train(self, X, y):
        
        #FUNÇÃO DE CUSTO: Otimiza o Recall com Ponderação de Classe
        def cost(params):
            try:
                n_estimators = max(1, int(params[0]))
                depth = max(1, int(params[1]))
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=depth, 
                    random_state=42, 
                    n_jobs=-1,
                    # AUMENTO DE REGULARIZAÇÃO: Impede folhas com poucas amostras, suavizando a previsão
                    min_samples_leaf=5,
                    # Ponderação de classe no PSO
                    class_weight='balanced' 
                )
                
                # Validação Cruzada usando RECALL
                scores = cross_val_score(model, X, y, cv=5, scoring='recall', n_jobs=-1, error_score='raise')
                
                # Retornamos 1 - Recall (para minimização)
                return 1 - np.mean(scores)
            except Exception as e:
                return 99999.0

        # 2. EXECUÇÃO DO PSO
        bounds = [(1.0, 100.0), (10.0, 50.0)] 
        pso_bounds = [np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])]

        print("Iniciando Otimização por PSO...")
        pso = PSO(func=cost, dim=2, bounds=pso_bounds, max_iter=5, num_particles=8)
        best_params = pso.optimize()

        best_n_estimators = max(1, int(best_params[0]))
        best_max_depth = max(10, int(best_params[1])) 
        
        self.best_params = {'n_estimators': best_n_estimators, 'max_depth': best_max_depth}
        print(f"PSO Otimizou Parâmetros: n_estimators={best_n_estimators}, max_depth={best_max_depth}")

        # TREINAMENTO FINAL COM SMOTE E PARÂMETROS OTIMIZADOS
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # APLICAÇÃO DO SMOTE APENAS NO CONJUNTO DE TREINO!
        print("Aplicando SMOTE para balanceamento...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # O Random Forest agora é treinado com os dados balanceados e parâmetros regularizados
        self.model = RandomForestClassifier(
            n_estimators=best_n_estimators,
            max_depth=best_max_depth,
            random_state=42,
            n_jobs=-1,
            # REGULARIZAÇÃO APLICADA NO TREINAMENTO FINAL
            min_samples_leaf=5,
            class_weight='balanced' 
        )
        self.model.fit(X_train_smote, y_train_smote)

        # AVALIAÇÃO DE DESEMPENHO E ARMAZENAMENTO DA IMPORTÂNCIA
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        self.feature_importances_ = self.model.feature_importances_
        
        return metrics

    def predict(self, X):
        # Retorna a PROBABILIDADE da classe 1 (Falha) para ranqueamento de risco
        return self.model.predict_proba(X)[:, 1]