import numpy as np
from functools import partial
from scipy.signal import find_peaks
from tqdm import tqdm

import random_growth_src.lib as lib

# Peak finding setting
prominence_factor = 0.02
n_bins = 20

# Simulation settings
n_steps = int(3e4)
n_runs = int(2e4)
t_max = 500
dt = t_max/n_steps

t_eval = 400
x0 = np.full(n_runs, np.log10(1))
eta0 = np.full(n_runs, 0.0)

# Parameter grid to search overs
grid_length = 20
M = 1
lamb_tildes = np.logspace(-1, 1, grid_length)
D_tildes = np.logspace(-0.5, 1.5, grid_length)
phase_diagram = np.zeros((len(D_tildes), len(lamb_tildes)))

total_sims = len(D_tildes) * len(lamb_tildes)
with tqdm(total=total_sims, desc="Building phase diagram") as pbar:
    for i, D in enumerate(D_tildes):
        for j, lamb in enumerate(lamb_tildes):
            
            #Run simulation
            sim_step = partial(lib.two_patch_x_coloured_noise_step, M=M, D=D, lamb=lamb)
            results_x_coloured_two_patch, results_t = lib.run_simulation(x0, t_max, n_runs, n_steps, sim_step, eta0=eta0)
            
            #Find number of peaks
            counts, bin_edges = np.histogram(results_x_coloured_two_patch[int(t_eval/dt)], bins=n_bins)
            peaks, properties = find_peaks(counts, prominence=np.max(counts)*prominence_factor)
            n_peaks = len(peaks)
            
            #Save number of peaks
            phase_diagram[i, j] = n_peaks == 1
            
            pbar.update(1)

np.savez(f"../../../data/numerical_results/numerical_phase_diagram_prominence_{prominence_factor}.npz", 
         D_tildes=D_tildes, lamb_tildes=lamb_tildes, phase_diagram=phase_diagram)
        