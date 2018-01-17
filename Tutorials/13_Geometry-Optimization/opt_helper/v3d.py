# Functions to compute properties of 3d vectors, including angles,
# torsions, out-of-plane angles.  Several return False if the operation
# cannot be completed numerically, as for example a torsion in which 3
# points are collinear.
import numpy as np
from math import sqrt, fabs, sin, acos, asin, fsum

TORS_ANGLE_LIM = 0.017
TORS_COS_TOL   = 1e-10
DOT_PARALLEL_LIMIT = 1.e-10

def norm(v):
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def dot(v1, v2, length=None):
    if length is None:
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    else:
        return fsum([v1[i] * v2[i] for i in range(length)])


def dist(v1, v2):
    return sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2 + (v2[2] - v1[2])**2)


# Normalize vector in place.  If norm exceeds thresholds, don't normalize and return False..
def normalize(v1, Rmin=1.0e-8, Rmax=1.0e15):
    n = norm(v1)
    if n < Rmin or n > Rmax:
        return False
    else:
        v1 /= n
        return True


def axpy(a, X, Y):
    Z = np.zeros(Y.shape, float)
    Z = a * X + Y
    return Z


# Compute and return normalized vector from point p1 to point p2.
# If norm is too small, don't normalize and return check as False.
def eAB(p1, p2):
    eAB = p2 - p1
    check = normalize(eAB)
    return check, eAB


# Compute and return cross-product.
def cross(u, v):
    X = np.zeros(3, float)
    X[0] = u[1] * v[2] - u[2] * v[1]
    X[1] = -u[0] * v[2] + u[2] * v[0]
    X[2] = u[0] * v[1] - u[1] * v[0]
    return X


# Are two vectors parallel?
def are_parallel(u, v):
    if fabs(dot(u, v) - 1.0e0) < DOT_PARALLEL_LIMIT: return True
    else: return False


# Are two vectors parallel?
def are_antiparallel(u, v):
    if fabs(dot(u, v) + 1.0e0) < DOT_PARALLEL_LIMIT: return True
    else: return False


def are_parallel_or_antiparallel(u, v):
    return are_parallel(u, v) or are_antiparallel(u, v)


# Compute and return angle in radians A-B-C (between vector B->A and vector B->C)
# If points are absurdly close or far apart, returns False
# tol is nearness of cos to 1/-1 to set angle 0/pi.
# Returns boolean check and value.
def angle(A, B, C, tol=1.0e-14):
    check, eBA = eAB(B, A)
    if not check:
        print("Warning: could not normalize eBA in angle()\n")
        return False, 0.0

    check, eBC = eAB(B, C)
    if not check:
        print("Warning: could not normalize eBC in angle()\n")
        return False, 0.0

    dotprod = dot(eBA, eBC)

    if dotprod > 1.0 - tol:
        phi = 0.0
    elif dotprod < -1.0 + tol:
        phi = acos(-1.0)
    else:
        phi = acos(dotprod)

    return True, phi


# Compute and return angle in dihedral angle in radians A-B-C-D
# returns false if bond angles are too large for good torsion definition
def tors(A, B, C, D):
    phi_lim = TORS_ANGLE_LIM
    tors_cos_tol   = TORS_COS_TOL

    # Form e vectors
    check1, EAB = eAB(A, B)
    check2, EBC = eAB(B, C)
    check3, ECD = eAB(C, D)

    if not check1 or not check2 or not check3:
        return False, 0.0

    # Compute bond angles
    check1, phi_123 = angle(A, B, C)
    check2, phi_234 = angle(B, C, D)

    if not check1 or not check2:
        return False, 0.0

    if phi_123 < phi_lim or phi_123 > (acos(-1) - phi_lim) or \
       phi_234 < phi_lim or phi_234 > (acos(-1) - phi_lim):
        return False, 0.0

    tmp = cross(EAB, EBC)
    tmp2 = cross(EBC, ECD)
    tval = dot(tmp, tmp2) / (sin(phi_123) * sin(phi_234))

    if tval >= 1.0 - tors_cos_tol:  # accounts for numerical leaking out of range
        tau = 0.0
    elif tval <= -1.0 + tors_cos_tol:
        tau = acos(-1)
    else:
        tau = acos(tval)

    # determine sign of torsion ; this convention matches Wilson, Decius and Cross
    if tau != acos(-1):  # no torsion will get value of -pi; Range is (-pi,pi].
        tmp = cross(EBC, ECD)
        tval = dot(EAB, tmp)
        if tval < 0:
            tau *= -1

    return True, tau


# Compute and return angle in dihedral angle in radians A-B-C-D
# returns false if bond angles are too large for good torsion definition
def oofp(A, B, C, D):
    check1, eBA = eAB(B, A)
    check2, eBC = eAB(B, C)
    check3, eBD = eAB(B, D)
    if not check1 or not check2 or not check3:
        return False, 0.0

    check1, phi_CBD = angle(C, B, D)
    if not check1:
        return False, 0.0

    # This shouldn't happen unless angle B-C-D -> 0,
    if sin(phi_CBD) < op.Params.v3d_tors_cos_tol:  #reusing parameter
        return False, 0.0

    dotprod = dot(cross(eBC, eBD), eBA) / sin(phi_CBD)

    if dotprod > 1.0: tau = acos(-1)
    elif dotprod < -1.0: tau = -1 * acos(-1)
    else: tau = asin(dotprod)
    return True, tau
