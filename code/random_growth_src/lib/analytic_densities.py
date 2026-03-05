import numpy as np
from functools import partial
from scipy.special import kv
from scipy.special import pbdv
from scipy.special import gamma
from scipy import integrate
from scipy.optimize import root_scalar

# White noise densities 

## N = 2

def two_patch_steady_state_density_x_white_noise(x, M, D):
        
    Z = 2 * kv(0, M / (2 * D))
    
    return 1 / Z * np.exp(- M / (2 * D) * np.cosh(x))

## N -> infinity

def mean_field_steady_state_density_y_white_noise(y, M, D):
    Z = gamma(M / D + 1) / (M / D) ** (M / D + 1)
    
    return 1 / Z * np.exp(- M /D * np.exp(-y) - (M / D + 1) * y)

def mean_field_steady_state_density_x_white_noise(x, M, D):
    
    Z = (gamma(M / D + 1) **2) / gamma(2 * (M /D + 1))
        
    return 1 / Z * (1 + np.exp(-x)) ** (- 2 * (M /D + 1)) * np.exp(- (M / D + 1) * x)

# Coloured noise densities

## N = 2

def unnormalized_two_patch_steady_state_density_x_coloured_noise(x, M, D, lamb):
    
    return (1 + M/lamb * np.cosh(x)) * np.exp(- M /(2 * D) * (1 + M / (2 * lamb) * np.cosh(x)) * np.cosh(x))

def partition_function_x_coloured_noise(M, D, lamb, x_range=200):
    
    Z = integrate.quad(unnormalized_two_patch_steady_state_density_x_coloured_noise, -x_range, x_range, args=(M, D, lamb))[0]
    
    return Z

def two_patch_steady_state_density_x_coloured_noise(x, M, D, lamb):
    
    Z = partition_function_x_coloured_noise(M, D, lamb)
    
    return 1 /Z * unnormalized_two_patch_steady_state_density_x_coloured_noise(x, M, D, lamb) 

## N -> infinity

def self_consistency_eqn(zeta, M, D, lamb):
    
    alpha = (M**2) / (2.0 * D * lamb)
    beta  = (M - (M / lamb) * (zeta + M)) / D
    nu = (zeta + M) / D
    
    z = beta / np.sqrt(2.0 * alpha)    
    Dv, Dv_prime = pbdv(-nu, z) 
    r1 = (z / 2.0) - (Dv_prime / Dv) 
    r2 = (z - r1) / (-nu)
    
    return np.sqrt(2.0 * alpha) / (nu - 1.0) * r1 + (M / lamb - 1.0) - M / lamb * (1.0 / np.sqrt(2.0 * alpha)) * nu * r2

def solve_zeta(M, D, lamb, eps=1e-12):

    sol = root_scalar(partial(self_consistency_eqn, M=M, D=D, lamb=lamb), method="brentq", bracket=(D * (1.0 + eps) - M, D), maxiter=10000)

    if (not sol.converged) or (not np.isfinite(sol.root)):
        raise RuntimeError(f"Root solve failed at D={D}, M={M}, lambda={lamb}")

    return sol.root

def weak_noise_approximation_for_zeta(M, D, lamb):
    
    M = np.asarray(M)
    D = np.asarray(D)
    lamb = np.asarray(lamb)

    return D * (lamb / (lamb + M))

def parabolic_cylinder(nu, beta):
    return pbdv(nu, beta)[0]

def modified_parabolic_cylinder(nu, alpha, beta):
    
    return (2 * alpha) ** (- nu / 2) * gamma(nu) * np.exp((beta ** 2) / (8 * alpha)) * parabolic_cylinder(-nu, beta / np.sqrt(2 * alpha))

def partition_function_y_coloured_noise(M, D, lamb, zeta):
    
    nu = (zeta + M) / D
    beta = (M - M / lamb * (zeta + M)) / D
    alpha = (M / lamb * M) / (2 * D)
    
    return modified_parabolic_cylinder(nu, alpha, beta) + M / lamb * modified_parabolic_cylinder(nu + 1, alpha, beta)

def mean_field_steady_state_density_y_coloured_noise(y, M, D, lamb, approximate_zeta=False):
    
    if not approximate_zeta:
        zeta = solve_zeta(M, D, lamb)
        
    else:
        zeta =  approximation_for_zeta(M, D, lamb)
    
    Z = partition_function_y_coloured_noise(M, D, lamb, zeta)
    
    return 1 / Z * (1 + M / lamb * np.exp(-y)) * np.exp(- 1 /D * ( (M**2 / lamb) / 2 * np.exp(-2 * y) + (M - M / lamb * (zeta + M)) * np.exp(-y) + (zeta + M) * y))

def mean_field_steady_state_density_x_coloured_noise(x, M, D, lamb, approximate_zeta=False):
    
    if not approximate_zeta:
        zeta = solve_zeta(M, D, lamb)
        
    else:
        zeta =  approximation_for_zeta(M, D, lamb)
    
    Z = partition_function_y_coloured_noise(M, D, lamb, zeta)
    
    nu = (zeta + M) / D
    beta = (M - M / lamb * (zeta + M)) / D
    alpha = (M / lamb * M) / (2 * D)
    
    A_x = alpha * (1 + np.exp(- 2 * x))
    B_x = beta * (1 + np.exp(-x))
    
    I1 = modified_parabolic_cylinder(2 * nu, A_x, B_x)
    I2 = modified_parabolic_cylinder(2 * nu + 1 , A_x, B_x)
    I3 = modified_parabolic_cylinder(2 * nu + 2, A_x, B_x)
    
    return (np.exp( - nu * x) / (Z ** 2)) * (I1 + M / lamb * (1 + np.exp(-x)) * I2 + ((M / lamb)**2 * np.exp(-x)) * I3)