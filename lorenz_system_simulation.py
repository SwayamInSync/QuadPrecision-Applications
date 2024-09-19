import numpy as np
from numpy_quaddtype import QuadPrecision, QuadPrecDType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def run_simulation(initial_state, num_steps, dt, dtype):
    states = np.zeros((num_steps, 3), dtype=dtype)
    states[0] = initial_state
    
    for i in range(1, num_steps):
        x, y, z = states[i-1]
        dx, dy, dz = lorenz(x, y, z)
        states[i] = states[i-1] + np.array([dx, dy, dz], dtype=dtype) * dtype(dt)
    
    return states

def run_simulation_128(initial_state, num_steps, dt, dtype):
    initial_state = np.array(initial_state, dtype=dtype)
    states = np.zeros((num_steps, 3), dtype=dtype)
    states[0] = initial_state
    
    for i in range(1, num_steps):
        x, y, z = states[i-1]
        dx, dy, dz = lorenz(x, y, z, QuadPrecision(10), QuadPrecision(28), QuadPrecision(2.667))
        states[i] = states[i-1] + np.array([dx, dy, dz], dtype=dtype) * QuadPrecision(str(dt))
    
    return states

def calculate_divergence(states1, states2):
    return np.sqrt(np.sum((states1 - states2)**2, axis=1))

def custom_cumsum(arr):
    """Custom cumulative sum function that preserves input dtype."""
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = result[i-1] + arr[i]
    return result

# Simulation parameters
num_steps = 1000000
dt = 0.0001
initial_state = [1.0, 1.0, 1.0]

# Small perturbation
epsilon = 1e-10
perturbed_state = [x + epsilon for x in initial_state]

# Run simulations
float64_states = run_simulation(initial_state, num_steps, dt, np.float64)
float64_perturbed = run_simulation(perturbed_state, num_steps, dt, np.float64)

quad_states = run_simulation_128(initial_state, num_steps, dt, QuadPrecDType("sleef"))
quad_perturbed = run_simulation_128(perturbed_state, num_steps, dt, QuadPrecDType("sleef"))

# Calculate divergence
float64_divergence = calculate_divergence(float64_states, float64_perturbed)
quad_divergence = calculate_divergence(quad_states, quad_perturbed)

# Plot divergence
plt.figure(figsize=(10, 6))
plt.loglog(float64_divergence, label='float64')
plt.loglog(quad_divergence, label='quad')

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# X coordinate over time
axs[0, 0].plot(float64_states[:, 0], label='float64 Original')
axs[0, 0].plot(float64_perturbed[:, 0], label='float64 Perturbed')
axs[0, 0].set_title('X Coordinate Over Time (float64)')
axs[0, 0].legend()

axs[0, 1].plot(quad_states[:, 0].astype(np.float64), label='quad Original')
axs[0, 1].plot(quad_perturbed[:, 0].astype(np.float64), label='quad Perturbed')
axs[0, 1].set_title('X Coordinate Over Time (quad)')
axs[0, 1].legend()

# Difference in X coordinate
float64_diff = np.abs(float64_states[:, 0] - float64_perturbed[:, 0])
quad_diff = np.abs(quad_states[:, 0] - quad_perturbed[:, 0])

axs[1, 0].semilogy(float64_diff, label='float64')
axs[1, 0].semilogy(quad_diff, label='quad')
axs[1, 0].set_title('Absolute Difference in X Coordinate (Log Scale)')
axs[1, 0].set_xlabel('Time Step')
axs[1, 0].set_ylabel('|X_original - X_perturbed|')
axs[1, 0].legend()

# Cumulative difference
axs[1, 1].plot(np.cumsum(float64_diff), label='float64')
axs[1, 1].plot(custom_cumsum(quad_diff), label='quad')
axs[1, 1].set_title('Cumulative Absolute Difference in X Coordinate')
axs[1, 1].set_xlabel('Time Step')
axs[1, 1].set_ylabel('Cumulative |X_original - X_perturbed|')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Final X difference (float64): {float64_diff[-1]}")
print(f"Final X difference (quad): {quad_diff[-1]}")
print(f"Time step where float64 X difference exceeds 1: {np.argmax(float64_diff > 1) * dt}")
print(f"Time step where quad X difference exceeds 1: {np.argmax(quad_diff > 1) * dt}")
