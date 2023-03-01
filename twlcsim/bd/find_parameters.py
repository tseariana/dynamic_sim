"""Routines for calculating the potential forces on the beads."""
import numpy as np


def find_parameters(length, b, num_beads):
    """Determine the parameters for the elastic forces based on ssWLC with twist."""
    sim_type = "sswlc"
    lp_dna = 53     # Persistence length of DNA in nm
    lt_dna = 110    # Twist persistence length of DNA in nm
    kuhn_length=b
    length_bead= length/(num_beads-1)
    length_dim = length_bead # * 0.34 / lp_dna        Non-dimensionalized length per bead
    param_values = np.loadtxt("bd/dssWLCparams")    # Load the parameter table from file

    # Determine the parameter values using linear interpolation of the parameter table
    eps_bend = np.interp(length_dim, param_values[:, 0], param_values[:, 1]) / length_dim
    gamma = np.interp(length_dim, param_values[:, 0], param_values[:, 2]) * length_dim
    # eps_par = np.interp(length_dim, param_values[:, 0], param_values[:, 3]) / length_dim
    eps_par = 3/(length_bead*kuhn_length**2)
    eps_perp = np.interp(length_dim, param_values[:, 0], param_values[:, 4]) / length_dim
    eta = np.interp(length_dim, param_values[:, 0], param_values[:, 5])
    xi_t3_drag_coef = np.interp(length_dim, param_values[:, 0], param_values[:, 6]) * length_dim
    eps_twist = lt_dna / (length_bead * 0.34)
    dt_bd = 0.01    #0.5 * xi_t3_drag_coef / (eps_perp * gamma**2)

    xi_r_drag_coef = length/num_beads       # Drag coefficient for position diffusion
    xi_t12_drag_coef = xi_t3_drag_coef    # Drag coefficient for orientation diffusion

    return sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta, \
        xi_r_drag_coef, xi_t3_drag_coef, xi_t12_drag_coef, dt_bd
