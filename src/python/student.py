import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


class MembraneModel:
    def __init__(self, c, M, x0, y0, modes, omegas_sq, beta, A, B):
        self.c = c
        self.M = M
        self.x0 = x0
        self.y0 = y0
        self.modes = modes
        self.omegas_sq = omegas_sq
        self.beta = beta
        self.A = A
        self.B = B

# writing our own versions of these functions to avoid scipy dependency
#here we are implenting the continuous algebraic riccati equation solver but we are using direct method
#we write the Hamiltonian matrix and then find its stable invariant subspace to compute the solution P
def solve_continuous_are(A, B, Q, R):
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    H = np.block([[A, -B @ R_inv @ B.T], [-Q, -A.T]])
    eigvals, eigvecs = np.linalg.eig(H)
    stable_indices = np.where(eigvals.real < -1e-10)[0]
    V = eigvecs[:, stable_indices]
    V1, V2 = V[:n, :], V[n:, :]
    P = (V2 @ np.linalg.inv(V1)).real
    return (P + P.T) / 2.0

#here we are writing our own version of the Runge-Kutta 4th order method to solve the initial value problem for the ODEs
#this is pretty basic implementation of fourth order Runge-Kutta method for solving ODEs, nothing fancy going on here
#just standard RK4 steps 
class OdeResult:
    def __init__(self, t, y):
        self.t = t; self.y = y

def solve_ivp(fun, t_span, y0, t_eval=None, **kwargs):
    if t_eval is None: t_eval = np.linspace(t_span[0], t_span[1], 100)
    y = np.zeros((len(y0), len(t_eval))); y[:, 0] = y0
    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i-1]
        t = t_eval[i-1]; yc = y[:, i-1]
        k1 = fun(t, yc)
        k2 = fun(t + dt/2, yc + dt/2 * k1)
        k3 = fun(t + dt/2, yc + dt/2 * k2)
        k4 = fun(t + dt, yc + dt * k3)
        y[:, i] = yc + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return OdeResult(t_eval, y)

# == Model construction and LQR design ==. copied from modal_lqr.py but with some edits to make it work without scipy dependency
def build_model(M=6, c=1.0, x0=0.37, y0=0.61, actuator="point", sigma=0.06, gamma=0.0):
    modes = [(m, n) for m in range(1, M + 1) for n in range(1, M + 1)]
    N = len(modes)
    
    # Eigenvalues and frequencies
    lams = np.array([np.pi**2 * (m*m + n*n) for m, n in modes])
    omegas_sq = (c**2) * lams

    # Coupling logic
    if actuator == "point":
        beta = np.array([2.0 * np.sin(m * np.pi * x0) * np.sin(n * np.pi * y0) for m, n in modes])
    else:
        # Simple trapezoidal integration for patch coupling
        grid = np.linspace(0.0, 1.0, 101)
        X, Y = np.meshgrid(grid, grid, indexing="ij")
        g = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2.0 * sigma**2))
        psi = g / np.trapz(np.trapz(g, x=grid, axis=1), x=grid)
        beta = []
        for m, n in modes:
            phi = 2.0 * np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
            beta.append(np.trapz(np.trapz(psi * phi, x=grid, axis=1), x=grid))
        beta = np.array(beta)

    # State Space A and B
    A = np.block([[np.zeros((N, N)), np.eye(N)], [-np.diag(omegas_sq), -gamma * np.eye(N)]])
    B = np.vstack([np.zeros((N, 1)), beta.reshape(N, 1)])
    
    return MembraneModel(c, M, x0, y0, modes, omegas_sq, beta, A, B)

def build_lqr(model, alpha=1.0, beta_v=1.0, R=5e-2):
    N = len(model.modes)
    Q = np.block([[alpha * np.diag(model.omegas_sq), np.zeros((N, N))], [np.zeros((N, N)), beta_v * np.eye(N)]])
    Rmat = np.array([[R]])
    P = solve_continuous_are(model.A, model.B, Q, Rmat)
    K = np.linalg.solve(Rmat, model.B.T @ P)
    return Q, Rmat, P, K

# == Simulation functions ==. again, we are writing our own versions of these to avoid scipy dependency
#in open loop we just solve the homogeneous system x' = Ax, and in closed loop we solve x' = (A - BK)x
def simulate_open_loop(model, x_init, T=6.0, nt=800):
    sol = solve_ivp(lambda t, x: model.A @ x, (0.0, T), x_init, np.linspace(0, T, nt))
    return sol.t, sol.y

def simulate_closed_loop(model, K, x_init, T=6.0, nt=800):
    def rhs(t, x):
        u = -(K @ x).item()
        return model.A @ x + model.B[:, 0] * u
    sol = solve_ivp(rhs, (0.0, T), x_init, np.linspace(0, T, nt))
    u_vals = np.array([-(K @ sol.y[:, i]).item() for i in range(sol.y.shape[1])])
    return sol.t, sol.y, u_vals