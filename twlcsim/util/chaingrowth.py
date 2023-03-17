import numpy as np


def random_walk_gauss_chain(length, beads, kuhn_length=1):
    r"""
    Generates random walk Gaussian chain
    :param length:   float
        Total chain length in kuhn lengths
    :param beads:   int
        Number of beads
    :param kuhn_length:   float
        default= 1
    :return r_poly: float array (beads x 3)
        Coordinates of beads
    """
    # initialize r_poly
    r_poly = np.zeros((beads, 3))

    # array of gaussian distributed step sizegis/distance between beads
    variance = (kuhn_length ** 2) * length / (beads - 1)
    r_poly_step = (1 / (np.sqrt(3))) * np.random.normal(loc=0, scale=np.sqrt(variance), size=(beads, 3))
    r_poly_step[0] = np.zeros((1, 3))  # maintain first bead at origin

    # cumulative sum of steps
    r_poly = np.cumsum(r_poly_step, axis=0)
    return r_poly

def gen_rouse_active(length_kuhn, num_beads, ka=1, fa=0, b=1, num_modes=10000):
    r"""
    Generate a discrete chain based on the active-Brownian Rouse model

    Parameters
    ----------
    length_kuhn : float
        Length of the chain (in Kuhn segments)
    num_beads : int
        Number of beads in the discrete chain
    ka : float
        Active force rate constant
    gamma : float
        Magnitude of the active forces
    b : float
        Kuhn length
    num_modes : int
        Number of Rouse modes in calculation

    Returns
    -------
    r_poly : (num_beads, 3) float
        Conformation of the chain subjected to active-Brownian forces

    """
    # Calculate the conformation, Brownian force, and active force
    r_poly = np.zeros((num_beads, 3))
    f_active = np.zeros((num_beads, 3))
    gamma= fa**2/ka

    ind = np.arange(num_beads)
    for p in range(1, num_modes + 1):
        sig_fp_tilde = np.sqrt(length_kuhn ** 2 * gamma * ka / p ** 2)
        ka_tilde = ka * length_kuhn ** 2 / p ** 2
        fp_tilde = np.random.randn(3) * sig_fp_tilde
        mu_xp = fp_tilde / (1 + ka_tilde)
        sig_xp = np.sqrt(1 + sig_fp_tilde ** 2 * ka_tilde / (1 + ka_tilde) ** 2)
        xp_tilde = (np.random.randn(3) * sig_xp + mu_xp)
        phi = np.sqrt(2) * np.cos(p * np.pi * ind / (num_beads - 1))
        r_poly += np.outer(phi, xp_tilde) * np.sqrt(length_kuhn) / p * (b/ np.sqrt(3*np.pi**2))
        f_active += np.outer(phi, fp_tilde) * p / np.sqrt(length_kuhn) * (np.sqrt(3*np.pi**2)/b)

    return r_poly, f_active


def confined_linear_chain(num_beads, a, fa=0):
    r"""
    Generates random walk Gaussian chain
    :param length:   float
        Total chain length in kuhn lengths
    :param beads:   int
        Number of beads
    :param kuhn_length:   float
        default= 1
    :return r_poly: float array (beads x 3)
        Coordinates of beads
    """
    # initialize r_poly
    r_poly = np.zeros((num_beads, 3))
    r_poly[:,0]= np.linspace(-a,a,num_beads)

    f_active = fa * np.random.randn(num_beads, 3)

    return r_poly, f_active

def confined_chain(num_beads, a, fa=0):
    r"""
    Generates random walk Gaussian chain
    :param length:   float
        Total chain length in kuhn lengths
    :param beads:   int
        Number of beads
    :param kuhn_length:   float
        default= 1
    :return r_poly: float array (beads x 3)
        Coordinates of beads
    """
    step_size = 1
    n_steps = num_beads
    R_confine = a

    r_theta = np.zeros((n_steps, 2))
    theta_start = np.random.rand() * np.pi * 2

    r_theta[0, :] = [R_confine, theta_start]

    for i in range(1, n_steps):

        d_edge = R_confine - r_theta[i - 1, 0]

        if d_edge > step_size:
            phi_max = np.pi
        elif d_edge == 0:
            phi_max = np.pi / 2 - np.arcsin((step_size / 2) / R_confine)
        else:
            phi_max = np.pi - np.arccos(d_edge / step_size)

        if i < n_steps - (R_confine / step_size):
            phi = 2 * (np.random.rand() - 0.5) * phi_max  # phi chosen from uniform distribution
        else:
            phi = ((-1 * np.abs(np.random.normal(0, i ** -0.5))) + 1) * phi_max * np.random.choice(
                [-1, 1])  # phi chosen from folded normal distribution
        r_prev = r_theta[i - 1, 0]

        r_theta[i, 0] = np.sqrt(r_prev * (r_prev - 2 * step_size * np.cos(phi)) + step_size ** 2)
        r_theta[i, 1] = np.arctan2(step_size * np.sin(phi), (r_prev - step_size * np.cos(phi))) + r_theta[i - 1, 1]

    xy = np.zeros((n_steps, 2))
    xy[:, 0] = r_theta[:, 0] * np.cos(r_theta[:, 1])
    xy[:, 1] = r_theta[:, 0] * np.sin(r_theta[:, 1])

    z = np.sqrt(np.abs(R_confine ** 2 - np.sum(xy ** 2, axis=1)))
    z_col = np.zeros(len(xy))
    z_col[0] = z[0]
    z_col[-1] = z[-1]
    for i in np.arange(1, len(z_col) - 1):
        while True:
            step = np.random.normal(step_size / np.sqrt(3), 1) * np.random.choice([-1, 1])
            proposed_z = z_col[i - 1] + step
            if (proposed_z > -1 * z[i] and proposed_z < z[i]):
                z_col[i] = proposed_z
                break

    r_poly = np.append(xy, z_col.reshape(-1, 1), axis=1)

    f_active = fa * np.random.randn(num_beads, 3)

    return r_poly, f_active