import numpy as np
from random_growth_src.lib import analytic_densities
from scipy import integrate
from scipy.special import kve

# White noise

def nd_two_patch_growth_rate_white_noise(M_tilde):
           
    return M_tilde * (kve(-1, M_tilde / 2)/kve(0, M_tilde / 2) - 1)

# Coloured noise

def nd_two_patch_log_steady_state_density_coloured_noise(x, M_tilde, lambda_tilde):
    
    return np.log1p(M_tilde / lambda_tilde * np.cosh(x)) - (M_tilde / 2) * (1 + M_tilde / lambda_tilde * np.cosh(x)) * np.cosh(x)

def nd_two_patch_steady_state_density_coloured_noise(x, M_tilde, lambda_tilde):
    
    return np.exp(nd_two_patch_log_steady_state_density_coloured_noise(x, M_tilde, lambda_tilde))

def nd_two_patch_log_partition_function_coloured_noise(M_tilde, lambda_tilde, xmin, xmax):
    
    m0 = nd_two_patch_log_steady_state_density_coloured_noise(0.0, M_tilde, lambda_tilde)

    def shifted_rho(x, M_tilde, lambda_tilde):
        return np.exp(nd_two_patch_log_steady_state_density_coloured_noise(x, M_tilde, lambda_tilde) - m0)

    Z_shift, err = integrate.quad(shifted_rho, xmin, xmax, args=(M_tilde, lambda_tilde),
                                 epsabs=1e-12, epsrel=1e-10, limit=200)
    
    return np.log(Z_shift) + m0

def nd_two_patch_growth_rate_coloured_noise(M_tilde, lambda_tilde, integration_width):
    
    xmin, xmax = -integration_width, integration_width
    m0 = nd_two_patch_log_steady_state_density_coloured_noise(0.0, M_tilde, lambda_tilde)

    def shifted_f(x, M_tilde, lambda_tilde):
        return np.exp(-x) * np.exp(nd_two_patch_log_steady_state_density_coloured_noise(x, M_tilde, lambda_tilde) - m0)

    integral_shift, err = integrate.quad(shifted_f, xmin, xmax, args=(M_tilde, lambda_tilde),
                                        epsabs=1e-12, epsrel=1e-10, limit=200)

    logI = np.log(integral_shift) + m0
    logZ = nd_two_patch_log_partition_function_coloured_noise(M_tilde, lambda_tilde, xmin, xmax)

    return M_tilde * np.expm1(logI - logZ)
    