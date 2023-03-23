"""
Monte Carlo and Brownian dynamics simulations of a discrete wormlike chain.

"""
from pathlib import Path
import numpy as np

from util.init_cond import init_cond
from bd.bd_sim import bd_sim

# --------------- Required for worm-like chain ------------------------
# from util.calc_writhe import calc_writhe
# from util.calc_twist import calc_twist
# ---------------------------------------------------------------------

def main():
    """Show example of BD simulation code usage."""

    # Initialize the simulation by reading parameters from file (located in the input directory)
    input_dir = "../input/"
    num_polymers = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[0]
    num_beads = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[1]
    length = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[2]
    b= np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[3]
    confinement_tag = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=bool)[4]
    confinement_radius = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[5]
    force_active0 = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[6]
    k_a = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[7]
    fl_only_tag = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=bool)[8]
    num_save_bd = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[9]
    time_save_bd = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[10]
    from_file = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=bool)[11]
    loaded_file = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=str)[12]
    output_dir = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=str)[13]

    if not from_file:
        loaded_file='0'

    assert num_polymers > 0
    assert num_beads > 0
    assert length > 0
    assert b > 0
    assert confinment_radius > 0
    assert num_save_bd > 0
    assert time_save_bd > 0
    
    # Initialize starting polymer configuration
    r_poly, t1_poly, t2_poly, t3_poly, twist_poly, force_active = init_cond(length, b, num_polymers, num_beads,
                                                                            force_active0, k_a, fl_only_tag,
                                                                            confinement_tag, confinement_radius, from_file,
                                                                            loaded_file, input_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # Perform the Brownian dynamics simulation
    for bd_count in range(1, num_save_bd + 1):
        r_poly, t1_poly, t2_poly, t3_poly, twist_poly, time_save_actual, force_active = bd_sim(r_poly, t1_poly, t2_poly, t3_poly, twist_poly,
                                                                                               num_polymers, num_beads, length, b,
                                                                                               confinement_tag, confinement_radius, force_active0, force_active, k_a,
                                                                                               fl_only_tag, time_save_bd)
        save_file(r_poly, t1_poly, t2_poly, t3_poly, force_active, bd_count, loaded_file, output_dir)

        # ------------Required for worm-like chain-------------------------------------------------------------
        # writhe = calc_writhe(r_poly, num_polymers, num_beads)
        # twist = calc_twist(twist_poly, num_polymers, num_beads)
        # -----------------------------------------------------------------------------------------------------

        print("Save point " + str(int(loaded_file)+bd_count) + " completed")


def save_file(r_poly, t1_poly, t2_poly, t3_poly, force_active, file_count, loaded_file, home_dir='.'):
    """
    Save the conformations to the output directory (output_dir).

    Saves the input *r_poly* to the file ``"r_poly_{file_count}"``, and
    saves each of the *ti_poly* to the file ``"t{i}_poly_{file_count}"``.

    Parameters
    ----------
    r_poly, t1_poly, t2_poly, t3_poly : (3, N) array_like
        The chain conformation information (to be saved).
    file_count : int
        A unique file index to append to the filename for this save point.
    home_dir : Path
    """
    home_dir = Path(home_dir)
    LABELS_position= np.array(([['Position', 'Position', 'Position', 't1', 't1', 't1', 't2', 't2', 't2','t3', 't3', 't3'],['x','y','z','x','y','z','x','y','z','x','y','z']]))
    LABELS_force= np.array(([['Active force', 'Active force', 'Active force'],['x', 'y','z']]))

    pos_file= np.hstack((r_poly,t1_poly,t2_poly,t3_poly))

    labeled_pos_file= np.vstack((LABELS_position,pos_file))
    labeled_force_file= np.vstack((LABELS_force,force_active))

    pos_name= 'pos_file_'+ str(int(loaded_file)+int(file_count))
    fa_name= 'fa_file_'+ str(int(loaded_file)+int(file_count))

    np.savetxt(home_dir/Path(pos_name), labeled_pos_file, delimiter=',', fmt='%s')
    np.savetxt(home_dir/Path(fa_name), labeled_force_file, delimiter= ',', fmt='%s')

if __name__ == "__main__":
    main()
