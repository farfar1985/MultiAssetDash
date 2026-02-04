"""
Quantum Simulator Ensemble Methods
==================================
Advanced quantum-inspired and quantum simulation approaches for model ensembling.

Methods:
1. Variational Quantum Eigensolver (VQE) for weight optimization
2. Quantum Approximate Optimization Algorithm (QAOA) 
3. Simulated Quantum Annealing (SQA) with transverse field
4. Grover-inspired adaptive search
5. Quantum Boltzmann Machine weights
6. Tensor Network ensemble (MPS/MPO inspired)
7. Quantum Walk ensemble selection

Author: AmiraB
Date: 2026-02-03
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.special import softmax
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try imports - graceful fallback if not available
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("PennyLane not available - using classical simulations")

try:
    import dimod
    from dimod import BinaryQuadraticModel
    DIMOD_AVAILABLE = True
except ImportError:
    DIMOD_AVAILABLE = False
    print("D-Wave dimod not available - using classical simulations")


class QuantumSimulatorEnsemble:
    """
    Collection of quantum-inspired ensemble methods using simulators.
    All methods work on classical hardware via simulation.
    """
    
    def __init__(self, n_models: int, random_state: int = 42):
        self.n_models = n_models
        self.rng = np.random.RandomState(random_state)
        self.weights = None
        self.history = []
    
    # =========================================================================
    # 1. VARIATIONAL QUANTUM EIGENSOLVER (VQE) INSPIRED OPTIMIZATION
    # =========================================================================
    
    def vqe_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                     n_layers: int = 4, n_iterations: int = 100) -> np.ndarray:
        """
        Use VQE-inspired parameterized circuit to find optimal weights.
        
        The "energy" we minimize is the prediction error (MSE).
        Parameters are rotation angles that determine model weights.
        """
        n_models = predictions.shape[1]
        
        # Initialize parameters (rotation angles)
        # More layers = more expressivity but harder to optimize
        n_params = n_models * n_layers
        params = self.rng.uniform(-np.pi, np.pi, n_params)
        
        def quantum_circuit_weights(params, n_models, n_layers):
            """
            Simulate a parameterized quantum circuit that outputs weights.
            
            Structure: Layer of Ry rotations → entangling → repeat
            Output: Squared amplitudes become weights
            """
            # Reshape params into layers
            params = params.reshape(n_layers, n_models)
            
            # Initialize state vector (all qubits in |0⟩)
            # We simulate the statevector
            state = np.zeros(2**n_models, dtype=complex)
            state[0] = 1.0  # |00...0⟩
            
            for layer in range(n_layers):
                # Apply Ry rotations (parameterized)
                for qubit in range(n_models):
                    angle = params[layer, qubit]
                    # Ry gate: rotation around Y-axis
                    c, s = np.cos(angle/2), np.sin(angle/2)
                    
                    # Apply to all basis states
                    new_state = np.zeros_like(state)
                    for idx in range(len(state)):
                        bit = (idx >> qubit) & 1
                        idx_flip = idx ^ (1 << qubit)
                        if bit == 0:
                            new_state[idx] += c * state[idx]
                            new_state[idx_flip] += s * state[idx]
                        else:
                            new_state[idx] += c * state[idx]
                            new_state[idx_flip] -= s * state[idx]
                    state = new_state
                
                # Apply entangling CZ gates (creates correlations)
                for qubit in range(n_models - 1):
                    for idx in range(len(state)):
                        if ((idx >> qubit) & 1) and ((idx >> (qubit + 1)) & 1):
                            state[idx] *= -1  # CZ: phase flip on |11⟩
            
            # Compute weights from measurement probabilities
            # Weight for model i = probability of measuring qubit i in |1⟩
            probs = np.abs(state)**2
            weights = np.zeros(n_models)
            for qubit in range(n_models):
                for idx in range(len(probs)):
                    if (idx >> qubit) & 1:
                        weights[qubit] += probs[idx]
            
            return weights / (weights.sum() + 1e-10)
        
        def cost_function(params):
            """MSE as the 'energy' to minimize"""
            weights = quantum_circuit_weights(params, n_models, n_layers)
            ensemble_pred = predictions @ weights
            mse = np.mean((ensemble_pred - actuals)**2)
            return mse
        
        # Optimize using gradient-free method (COBYLA like real VQE)
        result = minimize(cost_function, params, method='COBYLA',
                         options={'maxiter': n_iterations})
        
        self.weights = quantum_circuit_weights(result.x, n_models, n_layers)
        return self.weights
    
    # =========================================================================
    # 2. QAOA-INSPIRED ENSEMBLE SELECTION
    # =========================================================================
    
    def qaoa_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                      n_models_to_select: int = 10, p_rounds: int = 3) -> np.ndarray:
        """
        QAOA-inspired combinatorial optimization for model selection.
        
        Formulate as: Which K models minimize ensemble error?
        This is a subset selection problem → QUBO formulation.
        """
        n_models = predictions.shape[1]
        n_samples = len(actuals)
        
        # Build correlation matrix and individual errors
        errors = predictions - actuals.reshape(-1, 1)
        mse_individual = np.mean(errors**2, axis=0)
        
        # Covariance of errors (want diverse models)
        error_cov = np.cov(errors.T)
        if np.any(np.isnan(error_cov)):
            error_cov = np.eye(n_models)
        
        # QUBO formulation:
        # Minimize: sum_i (Q_ii * x_i) + sum_{i<j} (Q_ij * x_i * x_j)
        # where x_i ∈ {0,1} indicates model selection
        
        # Linear terms: prefer low individual error
        Q_diag = mse_individual
        
        # Quadratic terms: prefer uncorrelated models (diverse)
        # Positive cov = similar errors = bad → penalize
        Q_offdiag = error_cov.copy()
        np.fill_diagonal(Q_offdiag, 0)
        
        # Constraint: select exactly K models (penalty method)
        penalty = 10.0 * np.mean(mse_individual)
        
        def qaoa_cost(selection_probs):
            """
            Soft version of QUBO cost for gradient-based optimization.
            selection_probs are probabilities of selecting each model.
            """
            p = selection_probs
            
            # Linear cost
            linear = np.sum(Q_diag * p)
            
            # Quadratic cost (expected value)
            quadratic = p @ Q_offdiag @ p
            
            # Cardinality constraint: want exactly K
            k_violation = (np.sum(p) - n_models_to_select)**2
            
            return linear + 0.5 * quadratic + penalty * k_violation
        
        # QAOA optimization: alternate between mixer and cost Hamiltonians
        # We simulate this with alternating optimization
        
        # Initialize uniform
        gamma = self.rng.uniform(0, 2*np.pi, p_rounds)
        beta = self.rng.uniform(0, np.pi, p_rounds)
        probs = np.ones(n_models) / 2
        
        for round_idx in range(p_rounds):
            # Cost Hamiltonian evolution (gradient step toward lower cost)
            grad = np.zeros(n_models)
            eps = 1e-5
            for i in range(n_models):
                probs_plus = probs.copy()
                probs_plus[i] = min(1, probs[i] + eps)
                probs_minus = probs.copy()
                probs_minus[i] = max(0, probs[i] - eps)
                grad[i] = (qaoa_cost(probs_plus) - qaoa_cost(probs_minus)) / (2*eps)
            
            probs = probs - gamma[round_idx] * grad
            probs = np.clip(probs, 0, 1)
            
            # Mixer Hamiltonian (drive toward uniform superposition)
            probs = (1 - beta[round_idx]/np.pi) * probs + (beta[round_idx]/np.pi) * 0.5
        
        # Final: select top K by probability
        selected = np.argsort(probs)[-n_models_to_select:]
        
        # Compute weights for selected models (inverse MSE)
        weights = np.zeros(n_models)
        selected_mse = mse_individual[selected]
        selected_mse = np.maximum(selected_mse, 1e-10)
        selected_weights = 1.0 / selected_mse
        selected_weights /= selected_weights.sum()
        weights[selected] = selected_weights
        
        self.weights = weights
        return weights
    
    # =========================================================================
    # 3. SIMULATED QUANTUM ANNEALING (SQA)
    # =========================================================================
    
    def quantum_annealing_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                                    n_sweeps: int = 1000, n_trotter: int = 20) -> np.ndarray:
        """
        Simulated Quantum Annealing with transverse field Ising model.
        
        Unlike classical annealing, SQA uses quantum tunneling (via Trotter slices)
        to escape local minima more effectively.
        """
        n_models = predictions.shape[1]
        
        # Compute error matrix for QUBO formulation
        errors = predictions - actuals.reshape(-1, 1)
        mse = np.mean(errors**2, axis=0)
        cov = np.cov(errors.T)
        if np.any(np.isnan(cov)):
            cov = np.eye(n_models)
        
        # Ising model: h (local fields), J (couplings)
        # Map QUBO to Ising: x = (s+1)/2 where s ∈ {-1, +1}
        h = mse / 2  # Local field (bias toward selecting good models)
        J = cov / 4  # Coupling (penalize correlated model pairs)
        
        # Initialize Trotter slices (quantum replicas)
        # Each slice is a classical configuration
        spins = self.rng.choice([-1, 1], size=(n_trotter, n_models))
        
        # Annealing schedule
        T_start, T_end = 2.0, 0.01
        Gamma_start, Gamma_end = 3.0, 0.01  # Transverse field strength
        
        best_config = spins[0].copy()
        best_energy = float('inf')
        
        for sweep in range(n_sweeps):
            # Compute current temperature and transverse field
            progress = sweep / n_sweeps
            T = T_start * (T_end / T_start) ** progress
            Gamma = Gamma_start * (Gamma_end / Gamma_start) ** progress
            
            # Effective temperature for Trotter coupling
            J_perp = -T * np.log(np.tanh(Gamma / (n_trotter * T) + 1e-10)) / 2
            
            # Sweep through all spins in all Trotter slices
            for k in range(n_trotter):
                for i in range(n_models):
                    # Local field contribution
                    delta_E = 2 * spins[k, i] * h[i]
                    
                    # Coupling contribution (within slice)
                    for j in range(n_models):
                        if i != j:
                            delta_E += 2 * spins[k, i] * J[i, j] * spins[k, j]
                    
                    # Trotter coupling (between slices)
                    k_prev = (k - 1) % n_trotter
                    k_next = (k + 1) % n_trotter
                    delta_E += 2 * J_perp * spins[k, i] * (spins[k_prev, i] + spins[k_next, i])
                    
                    # Metropolis acceptance
                    if delta_E < 0 or self.rng.random() < np.exp(-delta_E / T):
                        spins[k, i] *= -1
            
            # Track best configuration found
            for k in range(n_trotter):
                energy = np.dot(h, (spins[k] + 1) / 2)
                energy += ((spins[k] + 1) / 2) @ J @ ((spins[k] + 1) / 2) / 2
                if energy < best_energy:
                    best_energy = energy
                    best_config = spins[k].copy()
        
        # Convert best Ising config to model selection
        selection = (best_config + 1) / 2  # Map {-1,+1} to {0,1}
        
        # Weight selected models by inverse MSE
        weights = np.zeros(n_models)
        selected_idx = selection > 0.5
        if selected_idx.any():
            inv_mse = 1.0 / (mse[selected_idx] + 1e-10)
            weights[selected_idx] = inv_mse / inv_mse.sum()
        else:
            weights = 1.0 / (mse + 1e-10)
            weights /= weights.sum()
        
        self.weights = weights
        return weights
    
    # =========================================================================
    # 4. GROVER-INSPIRED ADAPTIVE SEARCH
    # =========================================================================
    
    def grover_search_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                               error_threshold: float = None) -> np.ndarray:
        """
        Grover-inspired amplitude amplification for finding good models.
        
        Classically simulates the amplitude amplification effect:
        models with low error get their "amplitude" boosted quadratically.
        """
        n_models = predictions.shape[1]
        
        # Compute errors
        errors = np.mean((predictions - actuals.reshape(-1, 1))**2, axis=0)
        
        # Set threshold (auto if not specified)
        if error_threshold is None:
            error_threshold = np.median(errors)
        
        # "Good" models have error below threshold
        is_good = errors < error_threshold
        n_good = is_good.sum()
        
        if n_good == 0:
            # No models below threshold, use inverse error weighting
            weights = 1.0 / (errors + 1e-10)
            self.weights = weights / weights.sum()
            return self.weights
        
        # Grover's algorithm amplifies amplitude of good states quadratically
        # After ~sqrt(N/M) iterations, probability of good states approaches 1
        # where N = total states, M = good states
        
        # Simulate amplitude amplification
        # Initial amplitudes: uniform
        amplitudes = np.ones(n_models) / np.sqrt(n_models)
        
        # Optimal number of Grover iterations
        n_iterations = int(np.pi / 4 * np.sqrt(n_models / max(1, n_good)))
        n_iterations = max(1, min(n_iterations, 100))
        
        for _ in range(n_iterations):
            # Oracle: flip amplitude of good states
            amplitudes[is_good] *= -1
            
            # Diffusion operator: reflect about mean
            mean_amp = np.mean(amplitudes)
            amplitudes = 2 * mean_amp - amplitudes
        
        # Probabilities from amplitudes
        probs = amplitudes**2
        probs = np.maximum(probs, 0)  # Ensure non-negative
        
        # Weight by probability × inverse error
        weights = probs / (errors + 1e-10)
        self.weights = weights / weights.sum()
        return self.weights
    
    # =========================================================================
    # 5. QUANTUM BOLTZMANN MACHINE INSPIRED
    # =========================================================================
    
    def quantum_boltzmann_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                                   n_hidden: int = 5, n_samples: int = 1000) -> np.ndarray:
        """
        Quantum Boltzmann Machine inspired weight learning.
        
        Uses transverse field to enable tunneling between weight configurations.
        The visible layer represents model weights, hidden layer captures correlations.
        """
        n_visible = predictions.shape[1]
        
        # Initialize weights and biases
        W = self.rng.randn(n_visible, n_hidden) * 0.1
        b_v = np.zeros(n_visible)
        b_h = np.zeros(n_hidden)
        
        # Transverse field (enables quantum tunneling)
        Gamma = 1.0
        
        # Compute target: want weights that minimize prediction error
        errors = predictions - actuals.reshape(-1, 1)
        mse = np.mean(errors**2, axis=0)
        target = 1.0 / (mse + 0.01)  # Target activations: inverse error
        target /= target.sum()
        
        # Contrastive divergence with quantum tunneling
        learning_rate = 0.1
        
        for epoch in range(100):
            # Positive phase: clamp visible to target
            v_pos = target.copy()
            
            # Sample hidden given visible (with tunneling)
            h_prob = self._sigmoid(v_pos @ W + b_h + Gamma * self.rng.randn(n_hidden))
            h_pos = (self.rng.random(n_hidden) < h_prob).astype(float)
            
            # Negative phase: Gibbs sampling
            v_neg = v_pos.copy()
            h_neg = h_pos.copy()
            
            for _ in range(5):  # k-step CD
                # Quantum tunneling in visible layer
                tunnel_prob = Gamma * np.exp(-mse)
                tunnel = self.rng.random(n_visible) < tunnel_prob
                v_neg[tunnel] = target[tunnel]  # Tunnel to good states
                
                v_prob = self._sigmoid(h_neg @ W.T + b_v)
                v_neg = (1 - tunnel) * (self.rng.random(n_visible) < v_prob) + tunnel * v_neg
                
                h_prob = self._sigmoid(v_neg @ W + b_h)
                h_neg = (self.rng.random(n_hidden) < h_prob).astype(float)
            
            # Update weights
            W += learning_rate * (np.outer(v_pos, h_pos) - np.outer(v_neg, h_neg))
            b_v += learning_rate * (v_pos - v_neg)
            b_h += learning_rate * (h_pos - h_neg)
            
            # Decay transverse field
            Gamma *= 0.98
        
        # Final weights: expected visible activations
        weights = self._sigmoid(b_v)
        self.weights = weights / weights.sum()
        return self.weights
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    # =========================================================================
    # 6. TENSOR NETWORK (MPS) INSPIRED ENSEMBLE
    # =========================================================================
    
    def tensor_network_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                                bond_dim: int = 4) -> np.ndarray:
        """
        Matrix Product State (MPS) inspired model combination.
        
        Models are treated as a 1D chain. Correlations propagate through
        bond dimensions, capturing complex inter-model dependencies.
        """
        n_models = predictions.shape[1]
        n_samples = len(actuals)
        
        # Initialize MPS tensors
        # Each tensor A[i] has shape (bond_left, physical, bond_right)
        # Physical dimension = 2 (select or not)
        tensors = []
        for i in range(n_models):
            d_left = 1 if i == 0 else bond_dim
            d_right = 1 if i == n_models - 1 else bond_dim
            A = self.rng.randn(d_left, 2, d_right) * 0.1
            tensors.append(A)
        
        # Prepare cost function based on predictions
        errors = predictions - actuals.reshape(-1, 1)
        mse = np.mean(errors**2, axis=0)
        
        # DMRG-like optimization: sweep left and right
        for sweep in range(10):
            # Left sweep
            for i in range(n_models - 1):
                self._optimize_tensor_pair(tensors, i, mse, bond_dim)
            # Right sweep
            for i in range(n_models - 2, 0, -1):
                self._optimize_tensor_pair(tensors, i, mse, bond_dim)
        
        # Contract MPS to get probabilities
        probs = self._contract_mps(tensors)
        
        # Weight by probability × inverse error
        weights = probs / (mse + 1e-10)
        self.weights = weights / weights.sum()
        return self.weights
    
    def _optimize_tensor_pair(self, tensors, i, mse, bond_dim):
        """Optimize a pair of adjacent tensors"""
        # Local cost: selecting model i should reduce error
        # Simple gradient update
        grad = mse[i] - mse[i+1]  # Relative error
        
        # Adjust tensor to favor lower-error model
        tensors[i][:, 1, :] *= np.exp(-0.1 * mse[i])
        tensors[i+1][:, 1, :] *= np.exp(-0.1 * mse[i+1])
        
        # Renormalize
        for j in [i, i+1]:
            norm = np.linalg.norm(tensors[j])
            if norm > 1e-10:
                tensors[j] /= norm
    
    def _contract_mps(self, tensors):
        """Contract MPS to get selection probabilities"""
        n_models = len(tensors)
        
        # Contract from left
        result = tensors[0][:, 1, :]  # Select model 0
        probs = [np.sum(result**2)]
        
        for i in range(1, n_models):
            result = np.einsum('i,ijk->jk', result.flatten()[:tensors[i].shape[0]], tensors[i])
            probs.append(np.sum(result[:, 1, :]**2) if len(result.shape) > 2 else np.sum(result**2))
        
        probs = np.array(probs)
        probs = np.maximum(probs, 1e-10)
        return probs / probs.sum()
    
    # =========================================================================
    # 7. QUANTUM WALK ENSEMBLE SELECTION
    # =========================================================================
    
    def quantum_walk_ensemble(self, predictions: np.ndarray, actuals: np.ndarray,
                              n_steps: int = 50) -> np.ndarray:
        """
        Quantum walk on a graph of models for selection.
        
        Models are nodes, edges weighted by prediction correlation.
        Quantum walk spreads amplitude to well-connected, good models.
        """
        n_models = predictions.shape[1]
        
        # Build adjacency matrix from prediction correlations
        corr = np.corrcoef(predictions.T)
        corr = np.nan_to_num(corr, nan=0)
        
        # We want DIVERSE models, so use 1 - |corr| as edge weight
        adjacency = 1 - np.abs(corr)
        np.fill_diagonal(adjacency, 0)
        
        # Degree matrix
        degrees = adjacency.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
        
        # Normalized Laplacian for quantum walk
        L = np.eye(n_models) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        # Quantum walk Hamiltonian: H = γL + V
        # V is potential based on model quality
        errors = np.mean((predictions - actuals.reshape(-1, 1))**2, axis=0)
        V = np.diag(errors / (errors.max() + 1e-10))
        
        gamma = 0.5  # Hopping strength
        H = gamma * L + V
        
        # Initial state: uniform superposition
        psi = np.ones(n_models) / np.sqrt(n_models)
        
        # Time evolution: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        dt = 0.1
        for _ in range(n_steps):
            # Trotter approximation
            psi = psi - 1j * dt * (H @ psi)
            psi /= np.linalg.norm(psi)
        
        # Probability from amplitudes
        probs = np.abs(psi)**2
        
        # Weight by probability × inverse error
        weights = probs / (errors + 1e-10)
        self.weights = weights / weights.sum()
        return self.weights


def backtest_quantum_methods(predictions: np.ndarray, actuals: np.ndarray,
                             prices: np.ndarray, test_start: int = 200) -> pd.DataFrame:
    """
    Backtest all quantum simulator methods and compare performance.
    """
    results = []
    n_models = predictions.shape[1]
    
    qse = QuantumSimulatorEnsemble(n_models)
    
    methods = {
        'VQE': lambda p, a: qse.vqe_ensemble(p, a, n_layers=3, n_iterations=50),
        'QAOA': lambda p, a: qse.qaoa_ensemble(p, a, n_models_to_select=max(5, n_models//10)),
        'SQA': lambda p, a: qse.quantum_annealing_ensemble(p, a, n_sweeps=500),
        'Grover': lambda p, a: qse.grover_search_ensemble(p, a),
        'QBM': lambda p, a: qse.quantum_boltzmann_ensemble(p, a, n_hidden=5),
        'TensorNet': lambda p, a: qse.tensor_network_ensemble(p, a, bond_dim=4),
        'QuantumWalk': lambda p, a: qse.quantum_walk_ensemble(p, a, n_steps=30),
    }
    
    # Baseline: equal weight
    eq_weights = np.ones(n_models) / n_models
    
    for name, method_fn in methods.items():
        try:
            # Train on first portion, test on rest
            train_pred = predictions[:test_start]
            train_actual = actuals[:test_start]
            
            weights = method_fn(train_pred, train_actual)
            
            # Generate signals on test period
            test_pred = predictions[test_start:]
            test_prices = prices[test_start:]
            
            ensemble_pred = test_pred @ weights
            
            # Compute returns
            returns = np.diff(test_prices) / test_prices[:-1]
            signals = np.sign(ensemble_pred[:-1])
            strategy_returns = signals * returns
            
            # Metrics
            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)
            total_return = np.prod(1 + strategy_returns) - 1
            win_rate = np.mean(strategy_returns > 0)
            
            # Directional accuracy
            actual_direction = np.sign(returns)
            da = np.mean(signals == actual_direction)
            
            results.append({
                'Method': name,
                'Sharpe': round(sharpe, 3),
                'Return': f"{total_return*100:.1f}%",
                'WinRate': f"{win_rate*100:.1f}%",
                'DA': f"{da*100:.1f}%",
                'Active': int((weights > 0.01).sum()),
                'MaxWeight': f"{weights.max()*100:.1f}%"
            })
            
        except Exception as e:
            results.append({
                'Method': name,
                'Sharpe': 'ERROR',
                'Return': str(e)[:30],
                'WinRate': '-',
                'DA': '-',
                'Active': '-',
                'MaxWeight': '-'
            })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Quantum Simulator Ensemble Methods")
    print("=" * 50)
    print(f"PennyLane available: {PENNYLANE_AVAILABLE}")
    print(f"D-Wave dimod available: {DIMOD_AVAILABLE}")
    print("\nMethods implemented:")
    print("1. VQE (Variational Quantum Eigensolver)")
    print("2. QAOA (Quantum Approximate Optimization)")
    print("3. SQA (Simulated Quantum Annealing)")
    print("4. Grover-inspired Search")
    print("5. Quantum Boltzmann Machine")
    print("6. Tensor Network (MPS)")
    print("7. Quantum Walk")
