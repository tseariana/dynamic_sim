import numpy as np

def calc_boundary_force(r_poly, num_polymers, num_beads, a, epsilon):
    ## Initialize force boundary array
    num_total= num_beads*num_polymers
    force_boundary = np.zeros((num_total, 3))

    ## Calculate distance of beads from origin
    r = np.sqrt(np.sum(r_poly ** 2, axis=1))

    ## Find all beads with r greater than radius of boundary sphere
    indices= np.ndarray.flatten(np.argwhere(r>=a))

    ## Calculate force from boundary potential
    #boundary_potential[i]= epsilon/2*(r[i]-a)**12
    r_formatted =np.transpose(np.array([r,]*3))
    force_boundary[indices, :] = (-6 * epsilon * r_poly[indices, :] * (r_formatted[indices,:] - a) ** 11 )/ \
                                 r_formatted[indices,:]

    ## Pinning of first and last beads of each polymer
    first_bead_indices= np.arange(0,num_total, num_beads)
    last_bead_indices= np.arange(num_beads-1,num_total,num_beads)
    indices_f_l= np.append(first_bead_indices, last_bead_indices)
    force_boundary[indices_f_l, :] = (-6 * epsilon * r_poly[indices_f_l, :] *(r_formatted[indices_f_l,:] - a) ** 11) / \
                                     r_formatted[indices_f_l,:]

    return force_boundary



