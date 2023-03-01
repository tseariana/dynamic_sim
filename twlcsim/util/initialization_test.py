import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chaingrowth import random_walk_gauss_chain
from chaingrowth import gen_conf_rouse_active2
from chaingrowth import gen_conf_rouse_active1

"Generates confirmations using initialization codes and calculates statistics for verification"

def generate_init_config(length, num_beads, k_a, force_active0, num_config):
    r_poly = np.zeros((num_beads*num_config, 3))
    force_active = np.zeros((num_beads*num_config, 3))

    for n in np.arange(num_config):
        r_poly_n, force_active_n = gen_conf_rouse_active2(length_kuhn=length, num_beads=num_beads, ka=k_a, fa=force_active0,
                                                      b=1, num_modes=10000)
        r_poly[n * num_beads:(n + 1) * num_beads, :] = r_poly_n
        force_active[n * num_beads:(n + 1) * num_beads, :] = force_active_n
        if (n+1)%10==0:
            print(n+1, 'configurations generated')

    name = str('{0}_initialized_r_poly_configs'.format(num_config))
    np.savetxt(name, r_poly, delimiter=',')
    return name

def e2e_dis(file_name, num_config):
    r_poly = pd.read_csv(file_name, delimiter=",", header=None).to_numpy()
    num_total= len(r_poly)
    num_beads = num_total/num_config
    bead_f= np.arange(0, num_total, num_beads).astype(int)
    bead_l= np.arange(num_beads-1, num_total, num_beads).astype(int)
    e2e_vec= r_poly[bead_f,:]- r_poly[bead_l,:]
    e2e_sq_dis= np.sum(e2e_vec**2, axis=1)
    return e2e_sq_dis

def calc_normal_mode_amp(beads, r_vector, p):
    r"""
    Calculate the p_mode amplitude for a 3-dimensional vector of coordinates
    :param beads:   int
        Number of beads in the polymer chain
    :param r_vector:   float array (beads x 3)
        Array of the coordinates of the beads
    :param p:  int
        Value of p for the normal mode calculation
    :return Xp: float array (1 x 3)
        Amplitude of the p-th normal mode
    """
    beads=int(beads)
    if p == 0:
        phi = np.ones(beads)
    else:
        phi = np.sqrt(2) * np.cos((p * np.pi * np.arange(beads) / (beads - 1)))
    phi= np.reshape(phi, (-1,1))
    dn = 1 / (beads - 1)
    integrand = np.array(r_vector)*phi  # find xp at bead
    Xp= (np.sum(integrand, axis=0)- 0.5 * integrand[0] - 0.5 * integrand[beads-1]) * dn
    return Xp

def normal_mode(file_name, num_config, max_p):
    r_poly = pd.read_csv(file_name, delimiter=",", header=None).to_numpy()
    num_beads= len(r_poly)/num_config
    xp2_array= np.zeros((num_config, max_p+1))
    for n in np.arange(num_config):
        r_poly_n= r_poly[int(n*num_beads):int((n+1)*num_beads),:]
        for p in np.arange(max_p+1):
            xp= calc_normal_mode_amp(num_beads, r_poly_n, p)
            product= xp*xp
            xp2= np.sum(product)
            xp2_array[n, p]= xp2
    return xp2_array

def normal_mode_amp_theory(time, p, b, active_force_scale, ka, length):
    kp = ((3 * np.pi ** 2) / (b ** 2 * length)) * p ** 2
    tau_p = length ** 2 / (3 * np.pi ** 2 * p ** 2)
    t= time/tau_p
    theory_C= 3/kp *( np.exp(-t)+(active_force_scale**2*ka/(ka**2-(9*np.pi**4*p**4/length**4)))
        *(np.exp(-t)-(3*np.pi**2*p**2*np.exp(-ka*time)/(length**2*ka))))

    return theory_C

def normal_mode_analysis(file_name, num_config, num_p):
    print('Calculating normal mode amplitudes')
    xp2_array = normal_mode(file_name, num_config, num_p)
    config_array = np.arange(num_config)
    for p in np.arange(1, 11):
        plt.figure(0)
        plt.plot(config_array, xp2_array[:, p], alpha=0.6, label=str('p={0}'.format(int(p))))
        theory_C = normal_mode_amp_theory(0, p, 1, force_active0, k_a, length)
        plt.plot(config_array, theory_C* np.ones(config_array.shape), 'r--', alpha=0.6)
    plt.ylabel(r'$\langle \vec{X}_{p}(0)^{2} \rangle$')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.1, 1.15), loc='upper left')
    plt.savefig('normal_mode_amp2.png')
    ave_normal_mode_analysis(file_name, num_config, 10)
    return


def ave_normal_mode_analysis(file_name, num_config, num_p):
    xp2_array = normal_mode(file_name, num_config, num_p)
    p = np.arange(1, num_p+1)
    theory_C = normal_mode_amp_theory(0, p, 1, force_active0, k_a, length)
    xp2_average = np.average(xp2_array, axis=0)
    plt.figure(1)
    plt.plot(np.arange(1, num_p+1), xp2_average[1:], label='Simulation')
    plt.plot(np.arange(1, num_p+1), theory_C, label='Theory')
    plt.yscale('log')
    plt.xlabel('p')
    plt.ylabel('Xp**2')
    plt.legend()
    plt.savefig('normal_mode2_ave.png')


def e2e_analysis(file_name, num_config):
    print('Calculating e2e dis')
    config_array= np.arange(num_config)
    e2e_sq_dis = e2e_dis(file_name, num_config)
    plt.figure(2)
    plt.plot(config_array, e2e_sq_dis)
    plt.savefig('e2e_dis.png')
    return


def conduct_analysis(file_name, num_config):
    normal_mode_analysis(file_name, num_config, 20)
    e2e_analysis(file_name, num_config)
    ave_normal_mode_analysis(file_name, num_config, 20)
    return

def main():
    print('Generating configurations')
    file_name = generate_init_config(length, num_beads, k_a, force_active0, num_config)
    conduct_analysis(file_name, num_config)

length = 100
num_beads = 100
k_a = 1
force_active0 = 10
num_config= 1000

main()

#file_name=str('1000_initialized_r_poly_configs')
#num_config=1000
#ave_normal_mode_analysis(file_name, num_config, 10)





