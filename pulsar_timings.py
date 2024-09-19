import numpy as np
from numpy_quaddtype import QuadPrecDType, QuadPrecision
import matplotlib.pyplot as plt
from scipy import signal

def custom_arange(start, stop, dtype):
    if dtype == np.float64:
        return np.arange(start, stop, dtype=dtype)
    elif isinstance(dtype, QuadPrecDType):
        return np.array([QuadPrecision(str(i)) for i in range(int(start), int(stop))], dtype=dtype)
    else:
        raise ValueError("Unsupported dtype")

def simulate_pulsar_timing(num_years, precision='double'):
    num_days = num_years * 365
    rotation_period = 0.001  # 1 millisecond pulsar
    dtype = QuadPrecDType() if precision == 'quad' else np.float64
    
    time = custom_arange(0, num_days, dtype)
    true_phase = (time / QuadPrecision(str(rotation_period))) % QuadPrecision("1")
    
    # Increased red noise amplitude
    red_noise = np.cumsum(np.random.normal(0, 1e-8, num_days).astype(dtype))
    
    # Multiple gravitational wave effects
    gw_effect = QuadPrecision("0")
    for period, amp in [(5*365, 1e-7), (10*365, 5e-8), (15*365, 2e-8)]:
        gw_period = QuadPrecision(str(period))
        gw_amplitude = QuadPrecision(str(amp))
        gw_effect += gw_amplitude * np.sin(QuadPrecision("2") * np.pi * time / gw_period)
    
    # Add spin-down effect
    spin_down_rate = QuadPrecision("1e-15")  # Hz/s
    spin_down_effect = QuadPrecision("0.5") * spin_down_rate * time ** 2
    
    observed_phase = true_phase + red_noise + gw_effect + spin_down_effect
    
    # Decreased white noise amplitude
    toa_error = np.random.normal(0, 1e-9, num_days).astype(dtype)
    measured_toas = time + toa_error
    
    return measured_toas, observed_phase

def manual_dot(a, b):
    if a.ndim == 1 and b.ndim == 1:
        return sum(a[i] * b[i] for i in range(len(a)))
    elif a.ndim == 2 and b.ndim == 1:
        return np.array([sum(a[i][j] * b[j] for j in range(a.shape[1])) for i in range(a.shape[0])])
    elif a.ndim == 2 and b.ndim == 2:
        return np.array([[sum(a[i][k] * b[k][j] for k in range(a.shape[1])) for j in range(b.shape[1])] for i in range(a.shape[0])])
    else:
        raise ValueError("Unsupported array dimensions")

def manual_inv(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    n = matrix.shape[0]
    identity = np.eye(n, dtype=matrix.dtype)
    augmented = np.column_stack((matrix, identity))
    
    for i in range(n):
        pivot = augmented[i][i]
        augmented[i] = augmented[i] / pivot
        for j in range(n):
            if i != j:
                factor = augmented[j][i]
                augmented[j] -= factor * augmented[i]
    
    return augmented[:, n:]

def fit_timing_model(toas, phases, precision='double'):
    dtype = QuadPrecDType() if precision == 'quad' else np.float64
    ones = np.ones_like(toas).astype(dtype)
    X = np.column_stack([ones, toas, toas**2, np.sin(2*np.pi*toas/365), np.cos(2*np.pi*toas/365)])
    y = phases.astype(dtype)
    
    XTX = manual_dot(X.T, X)
    XTX_inv = manual_inv(XTX)
    XTy = manual_dot(X.T, y)
    
    beta = manual_dot(XTX_inv, XTy)
    predictions = manual_dot(X, beta)
    residuals = y - predictions
    
    return beta, residuals

def custom_fft(x):
    """A very basic FFT implementation for demonstration purposes."""
    n = len(x)
    if n <= 1:
        return x
    even = custom_fft(x[0::2])
    odd = custom_fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

def custom_periodogram(x, fs=1.0):
    """A simple periodogram implementation."""
    n = len(x)
    freqs = np.linspace(0, fs/2, n//2)
    X = custom_fft(x)
    psd = np.abs(np.array(X[:n//2]))**2 / (fs * n)
    return freqs, psd

def lomb_scargle(times, values, freqs):
    """A simple implementation of the Lomb-Scargle periodogram."""
    values_mean = np.mean(values)
    values = values - values_mean
    n = len(times)
    power = np.zeros_like(freqs)
    
    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        tau = np.arctan2(np.sum(np.sin(2 * omega * times)), np.sum(np.cos(2 * omega * times))) / (2 * omega)
        c = np.cos(omega * (times - tau))
        s = np.sin(omega * (times - tau))
        cc = np.sum(c * c)
        ss = np.sum(s * s)
        cs = np.sum(c * s)
        yc = np.sum(values * c)
        ys = np.sum(values * s)
        power[i] = (yc * yc / cc + ys * ys / ss) / np.sum(values * values)
    
    return power

def analyze_frequency(toas, residuals, dtype):
    # Convert to numpy arrays if they're not already
    toas = np.array(toas)
    residuals = np.array(residuals)
    
    # Calculate frequency range
    duration = toas[-1] - toas[0]
    min_freq = 1 / duration
    max_freq = 1 / (2 * np.mean(np.diff(toas)))  # Nyquist frequency
    freqs = np.logspace(np.float64(np.log10(min_freq)), np.float64(np.log10(max_freq)), num=1000).astype(dtype)
    
    power = lomb_scargle(toas, residuals, freqs)
    return freqs, power

def run_simulation():
    num_years = 100
    
    toas_double, phases_double = simulate_pulsar_timing(num_years, 'double')
    toas_quad, phases_quad = simulate_pulsar_timing(num_years, 'quad')
    
    beta_double, residuals_double = fit_timing_model(toas_double, phases_double, 'double')
    beta_quad, residuals_quad = fit_timing_model(toas_quad, phases_quad, 'quad')
    
    plt.figure(figsize=(12, 8))
    plt.plot(toas_double, residuals_double, label='Double Precision', alpha=0.7)
    plt.plot(toas_quad, residuals_quad, label='Quad Precision', alpha=0.7)
    plt.xlabel('Time (days)')
    plt.ylabel('Timing Residuals (s)')
    plt.title(f'Pulsar Timing Residuals over {num_years} Years')
    plt.legend()
    plt.savefig('pulsar_timing_residuals.png')
    plt.close()
    
    freqs_double, psd_double = analyze_frequency(toas_double, residuals_double, np.float64)
    freqs_quad, psd_quad = analyze_frequency(toas_quad, residuals_quad, QuadPrecDType)
    
    plt.figure(figsize=(12, 8))
    plt.semilogx(freqs_double, psd_double, label='Double Precision', alpha=0.7)
    plt.semilogx(freqs_quad, psd_quad, label='Quad Precision', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Frequency Analysis of Timing Residuals')
    plt.legend()
    plt.savefig('frequency_analysis.png')
    plt.close()

if __name__ == "__main__":
    run_simulation()
