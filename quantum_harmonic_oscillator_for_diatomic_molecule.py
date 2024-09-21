import numpy as np
import math
import matplotlib.pyplot as plt
from numpy_quaddtype import QuadPrecision, QuadPrecDType, pi
from matplotlib.ticker import LogLocator, LogFormatter


# Constants
h = QuadPrecision('6.62607015e-34')  # Planck's constant (J⋅s)
hbar = h / (QuadPrecision('2') * pi)  # Reduced Planck's constant
m_H = QuadPrecision('1.67353e-27')  # Mass of hydrogen atom (kg)
m_Cl = QuadPrecision('5.8862e-26')  # Mass of chlorine atom (kg)
k = QuadPrecision('480.0')  # Force constant for HCl (N/m)

# Calculated values
mu = (m_H * m_Cl) / (m_H + m_Cl)  # Reduced mass
omega = np.sqrt(k / mu)  # Angular frequency
f = omega / (QuadPrecision('2') * np.pi)  # Frequency

def energy_level(n, use_quad=True):
    if use_quad:
        return h * f * (n + QuadPrecision('0.5'))
    else:
        return np.float64(h) * np.float64(f) * (n + 0.5)

def hermite(n, x, use_quad=True):
    if use_quad:
        if n == 0:
            return QuadPrecision('1')
        elif n == 1:
            return QuadPrecision('2') * x
        else:
            h_prev = QuadPrecision('1')
            h_curr = QuadPrecision('2') * x
            for i in range(2, n + 1):
                h_next = QuadPrecision('2') * x * h_curr - QuadPrecision('2') * (i - 1) * h_prev
                h_prev = h_curr
                h_curr = h_next
            return h_curr
    else:
        if n == 0:
            return 1.0
        elif n == 1:
            return 2.0 * x
        else:
            h_prev = 1.0
            h_curr = 2.0 * x
            for i in range(2, n + 1):
                h_next = 2.0 * x * h_curr - 2.0 * (i - 1) * h_prev
                h_prev = h_curr
                h_curr = h_next
            return h_curr

def is_quad_inf(x):
    return x == QuadPrecision('inf') or x == QuadPrecision('-inf')

def is_quad_nan(x):
    return x != x  # NaN is the only value that doesn't equal itself

def wavefunction(n, x, use_quad=True):
    if use_quad:
        prefactor = QuadPrecision('1') / np.sqrt(QuadPrecision('2')**n * QuadPrecision(str(math.factorial(n))))
        gaussian = np.exp(-mu * omega * x**2 / (QuadPrecision('2') * hbar))
        hermite_poly = hermite(n, np.sqrt(mu * omega / hbar) * x)
        result = prefactor * (mu * omega / (np.pi * hbar))**QuadPrecision('0.25') * gaussian * hermite_poly
        
        if any(is_quad_inf(r) or is_quad_nan(r) for r in result):
            print(f"Invalid values in quad wavefunction for n={n}")
        return result
    else:
        prefactor = 1.0 / np.sqrt(2.0**n * math.factorial(n))
        gaussian = np.exp(-np.float64(mu * omega) * x**2 / (2.0 * np.float64(hbar)))
        hermite_poly = hermite(n, np.sqrt(np.float64(mu * omega / hbar)) * x, use_quad=False)
        result = prefactor * (np.float64(mu * omega) / (np.pi * np.float64(hbar)))**0.25 * gaussian * hermite_poly
        
        if np.any(np.isinf(result)) or np.any(np.isnan(result)):
            print(f"Invalid values in float64 wavefunction for n={n}")
        return result

def compare_and_plot(n_max):
    x = np.linspace(-5e-10, 5e-10, 10000, dtype=QuadPrecDType())
    dx = x[1] - x[0]
    
    quad_energy_diffs = []
    float_energy_diffs = []
    quad_norms = []
    float_norms = []
    theoretical_diff = h * f
    
    for n in range(n_max):
        # Energy difference calculations
        quad_diff = energy_level(n+1) - energy_level(n)
        float_diff = energy_level(n+1, use_quad=False) - energy_level(n, use_quad=False)
        
        quad_energy_diffs.append(abs(quad_diff - theoretical_diff))
        float_energy_diffs.append(abs(float_diff - np.float64(theoretical_diff)))
        
        # Wavefunction normalization calculations
        quad_psi = wavefunction(n, x)
        float_psi = wavefunction(n, x.astype(np.float64), use_quad=False)
        
        quad_norm = np.nansum(quad_psi**2) * dx
        float_norm = np.nansum(float_psi**2) * np.float64(dx)
        quad_norms.append(np.abs(quad_norm-1))
        float_norms.append(np.abs(float_norm-1))
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Energy difference plot
    ax1.semilogy(range(n_max), quad_energy_diffs, linestyle='--', label='Quad Precision', linewidth=2)
    ax1.semilogy(range(n_max), float_energy_diffs, label='Float64', linewidth=2)
    ax1.set_title('Absolute Error in Energy Level Differences from Theoretical (hf)')
    ax1.set_xlabel('Quantum Number n')
    ax1.set_ylabel('|E(n+1) - E(n) - hf| (J)')
    ax1.legend()
    ax1.grid(True)

    # Set more refined log ticks for y-axis
    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs='all', numticks=100))
    ax1.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    
    # Wavefunction normalization plot
    ax2.semilogy(range(n_max), quad_norms, linestyle='--', label='Quad Precision', linewidth=2)
    ax2.semilogy(range(n_max), float_norms, label='Float64', linewidth=2)
    ax2.set_title('Absolute Error in Wavefunction Normalization')
    ax2.set_xlabel('Quantum Number n')
    ax2.set_ylabel('|∫|ψ|²dx - 1|')
    ax2.legend()
    ax2.grid(True)

     # Set more refined log ticks for y-axis
    ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax2.yaxis.set_minor_locator(LogLocator(base=10, subs='all', numticks=100))
    ax2.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    
    plt.tight_layout()
    plt.savefig('quantum_oscillator_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run comparison and plotting
n_max = 150  # Increased to 100
compare_and_plot(n_max)
