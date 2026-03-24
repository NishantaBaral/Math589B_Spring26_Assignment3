# 
# In this module implement these two functions:
# 1. solve_continuous_are
# 2. solve_ivp
#
# Make sure that they are compatible with their usage
# in modal_lqr.py.
#
# The Gradescope Autograder will call your implementation
# through functions:
#
# 1. simulate_closed_loop
# 2. simulate_open_loop
import numpy as np


def solve_continuous_are(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solves the continuous-time Algebraic Riccati Equation (ARE) using the Direct Method.
    Returns the positive semi-definite matrix P.
    """
    # Get the state dimension
    n = A.shape[0] 
    
    # 1. Build the Hamiltonian Matrix (H)
    R_inv = np.linalg.inv(R)
    H_top_right = -B @ R_inv @ B.T
    
    H = np.block([
        [ A,  H_top_right],
        [-Q, -A.T        ]
    ])
    
    # 2. Compute all eigenvalues and eigenvectors of H
    # eigvals is a 1D array, eigvecs is a 2D array where columns are the vectors
    eigvals, eigvecs = np.linalg.eig(H)
    
    # 3. Find the Stable Eigenspace
    # We look for eigenvalues with strictly negative real parts.
    # We use -1e-10 instead of exactly 0 to account for floating-point rounding errors.
    stable_indices = np.where(eigvals.real < -1e-10)[0]
    
    # Sanity Check: A proper Hamiltonian should have exactly 'n' stable eigenvalues
    if len(stable_indices) != n:
        raise ValueError(f"Expected {n} stable eigenvalues, found {len(stable_indices)}. "
                         "Check your actuator location; the system might be uncontrollable!")
        
    # Extract the 'n' stable eigenvectors to form the matrix V
    V = eigvecs[:, stable_indices]
    
    # 4. Partition V into top half (V1) and bottom half (V2)
    V1 = V[:n, :]
    V2 = V[n:, :]
    
    # 5. Compute P = V2 * (V1)^-1
    # Because H is asymmetric, V1 and V2 contain complex numbers. 
    # However, the true solution P is guaranteed by the math to be purely real.
    P_complex = V2 @ np.linalg.inv(V1)
    
    # Discard the microscopic imaginary rounding errors
    P = P_complex.real
    
    # 6. Enforce strict symmetry
    # The true P must be symmetric. Numerical math can make it slightly off (e.g., P[0,1] != P[1,0])
    # Averaging it with its transpose forces it to be perfectly symmetric.
    P = (P + P.T) / 2.0
    
    return P
class OdeResult:
    """A simple class to mimic scipy's return object."""
    def __init__(self, t, y):
        self.t = t
        self.y = y

def solve_ivp(fun, t_span, y0, t_eval=None, **kwargs):
    """
    Custom initial value problem solver using the fixed-step Runge-Kutta 4 (RK4) method.
    """
    if t_eval is None:
        # Default to 100 steps if none provided
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        
    # Initialize the output array: rows = states, columns = time steps
    y = np.zeros((len(y0), len(t_eval)))
    y[:, 0] = y0
    
    # Runge-Kutta 4 Integration Loop
    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i-1]
        t = t_eval[i-1]
        y_curr = y[:, i-1]
        
        # Calculate the four RK4 slopes
        k1 = fun(t, y_curr)
        k2 = fun(t + dt/2.0, y_curr + (dt/2.0) * k1)
        k3 = fun(t + dt/2.0, y_curr + (dt/2.0) * k2)
        k4 = fun(t + dt, y_curr + dt * k3)
        
        # Combine slopes to step forward in time
        y[:, i] = y_curr + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
    return OdeResult(t_eval, y)
def simulate_open_loop(model, x_init: np.ndarray, T: float = 6.0, nt: int = 800):
    """
    Simulates the membrane vibrating naturally with no control input.
    """
    # Define the physics: the derivative of the state is just A * x
    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        return model.A @ x

    # Create the time array and pass it to YOUR custom solve_ivp
    t_eval = np.linspace(0.0, T, nt)
    sol = solve_ivp(rhs, (0.0, T), x_init, t_eval=t_eval)
    
    return sol.t, sol.y

def simulate_closed_loop(model, K: np.ndarray, x_init: np.ndarray, T: float = 6.0, nt: int = 800):
    """
    Simulates the membrane being actively damped by the LQR controller.
    """
    # Define the physics: derivative incorporates the control feedback
    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        # Calculate the instantaneous force required
        u = float(-(K @ x).item())
        # Apply the force to the B matrix
        return model.A @ x + model.B[:, 0] * u

    # Run the custom solver
    t_eval = np.linspace(0.0, T, nt)
    sol = solve_ivp(rhs, (0.0, T), x_init, t_eval=t_eval)
    
    # After the simulation, reconstruct exactly how hard the actuator pressed at every step
    controls = np.array([float(-(K @ sol.y[:, j]).item()) for j in range(sol.y.shape[1])])
    
    return sol.t, sol.y, controls