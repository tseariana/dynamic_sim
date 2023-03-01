"""Utilities for vector calculations."""
import numpy as np


def fix_t_vectors(t1_poly, t2_poly, t3_poly):
    """Ensure the orientation vectors are orthogonal and normalized."""
    # normalize t3 so that remove_perp works
    t3_poly /= np.linalg.norm(t3_poly, axis=1)[:, None]
    # subtract the perpendicular projection of t1 on t3
    dot = np.sum(t1_poly * t3_poly, axis=1)
    t1_poly = t1_poly - dot[:, None] * t3_poly
    # now t1 is pointing in the correct direction, normalize it
    t1_poly /= np.linalg.norm(t1_poly, axis=1)[:, None]
    # reset t2 to the cross product of t3 and t1
    t2_poly = np.cross(t3_poly, t1_poly)
    return t1_poly, t2_poly, t3_poly
