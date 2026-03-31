import numpy as np

# White noise simulations

def two_patch_x_white_noise_step(dt, x, eta, M, D):
    
    dWx = np.sqrt(4 * D * dt) * np.random.normal(size=len(x))
    x =  x + (- M * np.sinh(x) * dt + dWx)
        
    return x, 0

def mean_field_y_white_noise_step(dt, y, eta, M, D):
    
    dW = np.sqrt(2 * D * dt) * np.random.normal(size=len(y))
    y = y + ((M * (np.exp(-y) - 1) - D) * dt + dW)
    
    return y, 0

# Coloured noise simulations

def two_patch_x_coloured_noise_step(dt, x, eta, M, D, lamb):
    
    x = x + ((- M * np.sinh(x) + eta) * dt)
    
    a = np.exp(-lamb * dt)
    eta = a * eta + np.sqrt(2 * D * lamb * (1.0 - a*a)) * np.random.normal(size=len(eta))

    return x, eta

def mean_field_y_coloured_noise_step(dt, y, eta, M, D, lamb):
    
    zeta = np.mean(np.exp(y) * eta)
    y = y + ((M * (np.exp(-y) - 1.0) - zeta + eta) * dt)
    
    a = np.exp(-lamb * dt)
    eta = a * eta + np.sqrt(D * lamb * (1.0 - a*a)) * np.random.normal(size=len(eta))
    
    return y, eta

# Shot-noise simulations

def two_patch_x_shot_noise_step(dt, x, eta, M, D, lamb, beta):
    
    x = x + ((-M * np.sinh(x) + eta) * dt)

    # shot-noise parameters
    encounter_rate = lamb / beta
    A = np.sqrt(2.0 * D * lamb * beta) 

    a = np.exp(-lamb * dt)
    eta = a * eta
    
    #number of antigen encounters
    N_plus = np.random.poisson(encounter_rate  * dt, size=len(eta))
    N_minus = np.random.poisson(encounter_rate  * dt, size=len(eta))

    idx_plus = np.flatnonzero(N_plus)
    for i in idx_plus:
        u = np.random.uniform(0.0, dt, size=N_plus[i])
        eta[i] += np.sum(A * np.exp(-lamb * (dt - u)))

    idx_minus = np.flatnonzero(N_minus)
    for i in idx_minus:
        u = np.random.uniform(0.0, dt, size=N_minus[i])
        eta[i] -= np.sum(A * np.exp(-lamb * (dt - u)))

    return x, eta

# Simulation runner

def run_simulation(x0, t_max, n_runs, n_steps, simulation_step, t0=0, eta0=0):
    
    t = t0
    x = x0
    eta = eta0
    
    results_x = np.zeros((n_steps, n_runs))
    results_t = np.zeros(n_steps)
    
    results_t[0] = t
    results_x[0] = x
    dt = t_max/n_steps
    for i in range(n_steps-1):
        
        x, eta = simulation_step(dt, x, eta)
        t += dt
        
        results_x[i+1] = x
        results_t[i+1] = t
        
    return results_x, results_t