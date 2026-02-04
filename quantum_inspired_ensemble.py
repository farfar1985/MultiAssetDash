"""
Quantum-Inspired Ensemble Methods (Scalable)
=============================================
These methods capture the essence of quantum optimization WITHOUT
exponential state-space simulation. They scale O(n) or O(n^2).

Key insight: Quantum advantage comes from:
1. Superposition - exploring multiple solutions simultaneously
2. Entanglement - capturing correlations
3. Interference - amplifying good solutions
4. Tunneling - escaping local minima

We approximate these effects classically.

Author: AmiraB
Date: 2026-02-03
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import expm
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class QuantumInspiredEnsemble:
    """Scalable quantum-inspired ensemble methods."""
    
    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)
        self.weights = None
    
    # =========================================================================
    # 1. QUANTUM-INSPIRED EVOLUTIONARY ALGORITHM (QIEA)
    # =========================================================================
    
    def qiea_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                      pop_size: int = 50, generations: int = 100) -> np.ndarray:
        """
        Quantum-Inspired Evolutionary Algorithm.
        
        Uses quantum bits (qubits) represented as probability amplitudes.
        Updates via rotation gates based on fitness.
        
        Reference: Han & Kim (2002) "Genetic Quantum Algorithm"
        """
        n_models = predictions.shape[1]
        
        # Initialize population of quantum individuals
        # Each individual is a vector of (alpha, beta) pairs representing |psi> = alpha|0> + beta|1>
        # Stored as angles: theta, where alpha = cos(theta), beta = sin(theta)
        Q = self.rng.uniform(0, np.pi/2, (pop_size, n_models))  # Angles
        
        # Best solution tracking
        best_fitness = -np.inf
        best_solution = None
        
        def fitness(binary_solution):
            """Fitness = negative MSE (maximize)"""
            weights = binary_solution / (binary_solution.sum() + 1e-10)
            pred = predictions @ weights
            mse = np.mean((pred - actuals)**2)
            return -mse  # Negative because we maximize
        
        def observe(q_individual):
            """Collapse quantum state to binary"""
            probs = np.sin(q_individual)**2  # P(1)
            return (self.rng.random(n_models) < probs).astype(float)
        
        for gen in range(generations):
            # Observe each quantum individual to get binary solutions
            P = np.array([observe(q) for q in Q])
            
            # Evaluate fitness
            fitness_vals = np.array([fitness(p) for p in P])
            
            # Update best
            best_idx = np.argmax(fitness_vals)
            if fitness_vals[best_idx] > best_fitness:
                best_fitness = fitness_vals[best_idx]
                best_solution = P[best_idx].copy()
            
            # Quantum rotation gate update
            for i in range(pop_size):
                for j in range(n_models):
                    # Compare individual bit with best solution bit
                    x_ij = P[i, j]
                    b_j = best_solution[j] if best_solution is not None else 0
                    
                    # Rotation angle based on lookup table (simplified)
                    delta_theta = 0.01 * np.pi  # Small rotation
                    
                    if x_ij == 0 and b_j == 1:
                        # Rotate toward |1>
                        if Q[i, j] < np.pi/2:
                            Q[i, j] += delta_theta
                    elif x_ij == 1 and b_j == 0:
                        # Rotate toward |0>
                        if Q[i, j] > 0:
                            Q[i, j] -= delta_theta
        
        # Final weights from best solution
        if best_solution is not None:
            self.weights = best_solution / (best_solution.sum() + 1e-10)
        else:
            self.weights = np.ones(n_models) / n_models
        
        return self.weights
    
    # =========================================================================
    # 2. QUANTUM PARTICLE SWARM (QPSO)
    # =========================================================================
    
    def qpso_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                      n_particles: int = 30, iterations: int = 100) -> np.ndarray:
        """
        Quantum Particle Swarm Optimization.
        
        Particles move according to quantum probability clouds around attractors.
        Uses mean best position (quantum behavior) instead of velocity.
        
        Reference: Sun, Feng & Xu (2004)
        """
        n_models = predictions.shape[1]
        
        # Initialize particles (positions = weights)
        particles = self.rng.dirichlet(np.ones(n_models), n_particles)
        
        # Personal best and global best
        p_best = particles.copy()
        p_best_fitness = np.full(n_particles, -np.inf)
        g_best = particles[0].copy()
        g_best_fitness = -np.inf
        
        def fitness(weights):
            pred = predictions @ weights
            mse = np.mean((pred - actuals)**2)
            return -mse
        
        for it in range(iterations):
            # Contraction-expansion coefficient (decreases over time)
            beta = 1.0 - 0.5 * it / iterations  # 1.0 -> 0.5
            
            for i in range(n_particles):
                # Evaluate fitness
                f = fitness(particles[i])
                
                # Update personal best
                if f > p_best_fitness[i]:
                    p_best_fitness[i] = f
                    p_best[i] = particles[i].copy()
                
                # Update global best
                if f > g_best_fitness:
                    g_best_fitness = f
                    g_best = particles[i].copy()
            
            # Compute mean best position (quantum center)
            m_best = np.mean(p_best, axis=0)
            
            for i in range(n_particles):
                # Local attractor: random point between personal and global best
                phi = self.rng.random(n_models)
                p = phi * p_best[i] + (1 - phi) * g_best
                
                # Quantum behavior: sample from probability cloud
                u = self.rng.random(n_models)
                sign = 2 * (self.rng.random(n_models) > 0.5) - 1
                
                # Position update (quantum)
                particles[i] = p + sign * beta * np.abs(m_best - particles[i]) * np.log(1 / u)
                
                # Ensure valid weights (positive, sum to 1)
                particles[i] = np.maximum(particles[i], 0)
                particles[i] /= particles[i].sum() + 1e-10
        
        self.weights = g_best
        return self.weights
    
    # =========================================================================
    # 3. SIMULATED BIFURCATION (SB)
    # =========================================================================
    
    def simulated_bifurcation_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                                       n_steps: int = 1000, dt: float = 0.1) -> np.ndarray:
        """
        Simulated Bifurcation Machine.
        
        Solves Ising problems by simulating classical Hamiltonian dynamics
        near a bifurcation point. Achieves quantum-like exploration.
        
        Reference: Goto et al. (2019)
        """
        n_models = predictions.shape[1]
        
        # Build Ising model from prediction correlations
        errors = predictions - actuals.reshape(-1, 1)
        mse = np.mean(errors**2, axis=0)
        cov = np.corrcoef(errors.T)
        cov = np.nan_to_num(cov, nan=0)
        
        # Ising parameters: want to select low-error, diverse models
        h = mse / mse.max()  # Local fields (penalty for high error)
        J = -0.5 * cov  # Couplings (negative = want uncorrelated)
        
        # Initialize position and momentum
        x = self.rng.randn(n_models) * 0.1
        y = np.zeros(n_models)
        
        # Kerr coefficient (nonlinearity)
        K = 1.0
        
        # Pump amplitude schedule
        p0 = 0.0
        
        for step in range(n_steps):
            # Increase pump amplitude (drives system through bifurcation)
            p = p0 + (1.0 - p0) * step / n_steps
            
            # Hamiltonian dynamics (symplectic integrator)
            # dx/dt = (p - K*x^2) * x - epsilon * dH/dy
            # dy/dt = epsilon * dH/dx
            
            # Gradient of Ising energy
            grad_x = h + J @ np.tanh(x)
            
            # Update (leapfrog-like)
            y_half = y + 0.5 * dt * grad_x
            x_new = x + dt * ((p - K * x**2) * x + 0.1 * y_half)
            y = y_half + 0.5 * dt * (h + J @ np.tanh(x_new))
            x = x_new
        
        # Convert continuous x to binary selection
        selection = (np.tanh(x) + 1) / 2  # Soft selection [0, 1]
        
        # Weight by selection × inverse error
        weights = selection / (mse + 0.01)
        self.weights = weights / (weights.sum() + 1e-10)
        return self.weights
    
    # =========================================================================
    # 4. COHERENT ISING MACHINE (CIM) SIMULATION
    # =========================================================================
    
    def cim_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                     n_rounds: int = 50) -> np.ndarray:
        """
        Coherent Ising Machine simulation.
        
        Models optical parametric oscillator (OPO) network solving Ising problems.
        Exploits amplitude-phase dynamics for optimization.
        
        Reference: Inagaki et al. (2016)
        """
        n_models = predictions.shape[1]
        
        # Build Ising coupling from error covariance
        errors = predictions - actuals.reshape(-1, 1)
        mse = np.mean(errors**2, axis=0)
        cov = np.corrcoef(errors.T)
        cov = np.nan_to_num(cov, nan=0)
        
        J = -cov  # Want diverse models
        h = mse / mse.max()  # Prefer low-error models
        
        # OPO amplitudes (complex field)
        c = self.rng.randn(n_models) * 0.01 + 1j * self.rng.randn(n_models) * 0.01
        
        # Pump schedule
        pump_schedule = np.linspace(0.1, 1.2, n_rounds)
        
        for r, pump in enumerate(pump_schedule):
            # Saturation nonlinearity
            sat = 1.0 / (1 + np.abs(c)**2)
            
            # OPO dynamics: c_dot = (pump - 1) * c - |c|^2 * c + J @ c
            noise = 0.01 * (self.rng.randn(n_models) + 1j * self.rng.randn(n_models))
            c = (pump - 1) * c * sat + 0.1 * (J @ np.real(c) - 1j * h) + noise
            
            # Amplitude limitation
            c = np.clip(np.real(c), -2, 2) + 1j * np.clip(np.imag(c), -2, 2)
        
        # Convert to weights: positive real part = selected
        selection = np.maximum(np.real(c), 0)
        weights = selection / (mse + 0.01)
        self.weights = weights / (weights.sum() + 1e-10)
        return self.weights
    
    # =========================================================================
    # 5. QUANTUM ANNEALING APPROXIMATION (EFFICIENT)
    # =========================================================================
    
    def efficient_qa_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                              n_sweeps: int = 100) -> np.ndarray:
        """
        Efficient quantum annealing approximation.
        
        Uses mean-field approximation instead of full Trotter simulation.
        O(n^2) instead of O(2^n).
        """
        n_models = predictions.shape[1]
        
        # Build QUBO
        errors = predictions - actuals.reshape(-1, 1)
        mse = np.mean(errors**2, axis=0)
        cov = np.corrcoef(errors.T)
        cov = np.nan_to_num(cov, nan=0)
        
        Q_diag = mse
        Q_off = cov
        
        # Mean-field spins (soft variables in [0, 1])
        m = np.ones(n_models) * 0.5
        
        # Annealing schedule
        T_start, T_end = 2.0, 0.01
        Gamma_start, Gamma_end = 2.0, 0.01
        
        for sweep in range(n_sweeps):
            progress = sweep / n_sweeps
            T = T_start * (T_end / T_start) ** progress
            Gamma = Gamma_start * (Gamma_end / Gamma_start) ** progress
            
            # Mean-field update
            for i in range(n_models):
                # Effective field on spin i
                h_eff = Q_diag[i] + np.sum(Q_off[i, :] * m) - Q_off[i, i] * m[i]
                
                # Quantum transverse field contribution
                # In mean-field: adds fluctuations
                h_eff += Gamma * (self.rng.randn() * 0.1)
                
                # Update mean-field variable
                m[i] = 1 / (1 + np.exp(h_eff / T))
        
        # Final weights
        weights = m / (mse + 0.01)
        self.weights = weights / (weights.sum() + 1e-10)
        return self.weights
    
    # =========================================================================
    # 6. VARIATIONAL QUANTUM-INSPIRED (VQI) 
    # =========================================================================
    
    def vqi_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                     n_layers: int = 3) -> np.ndarray:
        """
        Variational Quantum-Inspired optimization.
        
        Uses parameterized trigonometric functions instead of quantum gates.
        Captures rotational structure without exponential state space.
        """
        n_models = predictions.shape[1]
        
        # Parameters: 2 per model per layer (like Rx, Rz rotations)
        n_params = 2 * n_models * n_layers
        params = self.rng.uniform(-np.pi, np.pi, n_params)
        
        def param_to_weights(params):
            """Convert parameters to weights using trigonometric circuit."""
            params = params.reshape(n_layers, 2, n_models)
            
            # Start with uniform
            w = np.ones(n_models) / n_models
            
            for layer in range(n_layers):
                theta = params[layer, 0, :]  # Rotation angles
                phi = params[layer, 1, :]    # Phase angles
                
                # Apply "rotation" transformation
                w = w * np.cos(theta)**2 + np.roll(w, 1) * np.sin(theta)**2
                
                # Apply "entangling" transformation (neighbor mixing)
                w_new = 0.8 * w + 0.1 * np.roll(w, 1) + 0.1 * np.roll(w, -1)
                w = w_new * np.abs(np.cos(phi))
                
                # Normalize
                w = np.maximum(w, 1e-10)
                w /= w.sum()
            
            return w
        
        def cost(params):
            weights = param_to_weights(params)
            pred = predictions @ weights
            mse = np.mean((pred - actuals)**2)
            return mse
        
        # Optimize
        result = minimize(cost, params, method='L-BFGS-B',
                         options={'maxiter': 100})
        
        self.weights = param_to_weights(result.x)
        return self.weights
    
    # =========================================================================
    # 7. QUANTUM WALK INSPIRED SELECTION
    # =========================================================================
    
    def qwalk_selection_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                                 n_steps: int = 20) -> np.ndarray:
        """
        Quantum walk inspired model selection.
        
        Continuous-time quantum walk on model graph.
        Uses efficient matrix exponential (not full state vector).
        """
        n_models = predictions.shape[1]
        
        # Build graph: models connected by prediction similarity
        corr = np.corrcoef(predictions.T)
        corr = np.nan_to_num(corr, nan=0)
        
        # Adjacency: want diverse models, so use 1 - |corr|
        A = 1 - np.abs(corr)
        np.fill_diagonal(A, 0)
        
        # Laplacian
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Add potential based on model quality
        errors = np.mean((predictions - actuals.reshape(-1, 1))**2, axis=0)
        V = np.diag(errors / errors.max())
        
        # Hamiltonian
        gamma = 0.3
        H = gamma * L + V
        
        # Initial probability distribution (uniform)
        p = np.ones(n_models) / n_models
        
        # Time evolution of probability
        # In quantum walk: p(t) ~ |e^(-iHt) p(0)|^2
        # We use the classical analog: p(t) = e^(-Ht) p(0) / Z
        for step in range(n_steps):
            t = (step + 1) * 0.1
            # Approximate evolution
            p = p - 0.1 * (H @ p)
            p = np.maximum(p, 0)
            p /= p.sum() + 1e-10
        
        # Weight by walk probability × inverse error
        weights = p / (errors + 0.01)
        self.weights = weights / weights.sum()
        return self.weights


def benchmark_qi_methods(X: np.ndarray, y: np.ndarray, test_start: int = None) -> pd.DataFrame:
    """Benchmark all quantum-inspired methods."""
    if test_start is None:
        test_start = int(len(y) * 0.6)
    
    X_train, y_train = X[:test_start], y[:test_start]
    X_test, y_test = X[test_start:], y[test_start:]
    
    qie = QuantumInspiredEnsemble(random_state=42)
    
    methods = [
        ('QIEA', lambda: qie.qiea_ensemble(X_train, y_train, pop_size=30, generations=50)),
        ('QPSO', lambda: qie.qpso_ensemble(X_train, y_train, n_particles=20, iterations=50)),
        ('SimBifurcation', lambda: qie.simulated_bifurcation_ensemble(X_train, y_train, n_steps=500)),
        ('CIM', lambda: qie.cim_ensemble(X_train, y_train, n_rounds=30)),
        ('EfficientQA', lambda: qie.efficient_qa_ensemble(X_train, y_train, n_sweeps=100)),
        ('VQI', lambda: qie.vqi_ensemble(X_train, y_train, n_layers=3)),
        ('QWalk', lambda: qie.qwalk_selection_ensemble(X_train, y_train, n_steps=20)),
    ]
    
    results = []
    
    for name, fn in methods:
        try:
            import time
            start = time.time()
            weights = fn()
            elapsed = time.time() - start
            
            # Backtest
            pred = X_test @ weights
            signal = np.sign(pred)
            returns = np.diff(y_test) / y_test[:-1]
            signal = signal[:len(returns)]
            
            active = signal != 0
            if active.any():
                strat_ret = signal[active] * returns[active]
                sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
                total_ret = np.prod(1 + strat_ret) - 1
                win_rate = np.mean(strat_ret > 0)
                da = np.mean(signal[active] == np.sign(returns[active]))
            else:
                sharpe = total_ret = win_rate = da = 0
            
            results.append({
                'Method': name,
                'Sharpe': round(sharpe, 3),
                'Return': f"{total_ret*100:.1f}%",
                'WinRate': f"{win_rate*100:.1f}%",
                'DA': f"{da*100:.1f}%",
                'Active': int((weights > 0.01).sum()),
                'Time': f"{elapsed:.1f}s"
            })
            
        except Exception as e:
            results.append({
                'Method': name,
                'Sharpe': 'ERR',
                'Return': str(e)[:30],
                'WinRate': '-',
                'DA': '-',
                'Active': '-',
                'Time': '-'
            })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Quantum-Inspired Ensemble Methods (Scalable)")
    print("=" * 50)
    print("\nMethods implemented (all O(n) or O(n^2)):")
    print("1. QIEA - Quantum-Inspired Evolutionary Algorithm")
    print("2. QPSO - Quantum Particle Swarm Optimization")
    print("3. SimBifurcation - Simulated Bifurcation Machine")
    print("4. CIM - Coherent Ising Machine simulation")
    print("5. EfficientQA - Mean-field Quantum Annealing")
    print("6. VQI - Variational Quantum-Inspired")
    print("7. QWalk - Quantum Walk Selection")
