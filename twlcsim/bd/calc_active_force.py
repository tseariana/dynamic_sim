import numpy as np

def calc_temp_corr_active_force(num_total, force_active0, k_a, dt_bd, force_active, fl_only_tag, fl_indices):
    """
    Calculate temporally-correlated active force based off the theory of Ghosh et al (2022).

    :param num_total: Number of beads in simulation
    :param force_active0: Active force scale
    :param k_a: Active force relaxation rate
    :param dt_bd: Differential time step
    :param force_active: Previously applied active force
    :param fl_only_tag: boolean -- whether active force is only applied on first and last beads of chains
    :param fl_indices: Indices of first and last beads of chain

    :return: dt_bd_a: Differential time step for active force
    :return: mag_active: Active force magnitude
    :return: w0: variable required for proper integration of active force
    :return: active_force_r_integration: Active force integration
    """
    dt_bd_a = dt_bd * k_a
    mag_active = np.sqrt(2 * force_active0 ** 2 * k_a)

    w0_2 = 1 / (2 * k_a) * (1 - np.exp(-2 * dt_bd_a))
    w1_2 = 1 / (2 * k_a ** 3) * (2 * dt_bd_a - 3 - np.exp(-2 * dt_bd_a) + 4 * np.exp(-dt_bd_a))
    w0w1 = 1 / (2 * k_a ** 2) * (1 - 2 * np.exp(-dt_bd_a) + np.exp(-2 * dt_bd_a))

    # active force calculation
    w0 = np.sqrt(w0_2) * np.random.randn(num_total, 3)
    w1 = w0w1 / np.sqrt(w0_2) * np.random.randn(num_total, 3) + np.sqrt(np.absolute(w1_2 - (w0w1 ** 2 / w0_2))) * np.random.randn(num_total, 3)

    active_force_r_integration = (1 / k_a * (1 - np.exp(-dt_bd_a)) * force_active + mag_active * w1)

    if fl_only_tag:
        active_force_r_integration_0= np.zeros(active_force_r_integration.shape)
        active_force_r_integration_0[fl_indices,:] = active_force_r_integration[fl_indices,:]
        active_force_r_integration= active_force_r_integration_0
    else:
        pass

    return dt_bd_a, mag_active, w0, active_force_r_integration

