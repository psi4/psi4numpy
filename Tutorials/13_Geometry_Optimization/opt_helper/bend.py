from math import sqrt, cos

import numpy as np

from . import covRadii
from . import optExceptions
from . import v3d
from .misc import delta, HguessLindhRho
from .simple import *

from psi4 import constants
BOHR2ANGSTROMS = constants.bohr2angstroms
HARTREE2AJ = constants.hartree2aJ

class BEND(SIMPLE):
    def __init__(self, a, b, c, frozen=False, fixedEqVal=None, bendType="REGULAR"):

        if a < c: atoms = (a, b, c)
        else: atoms = (c, b, a)

        self.bendType = bendType
        self._axes_fixed = False
        self._x = np.zeros(3, float)
        self._w = np.zeros(3, float)

        SIMPLE.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen: s = '*'
        else: s = ' '

        if self.bendType == "REGULAR":
            s += "B"
        elif self.bendType == "LINEAR":
            s += "L"
        elif self.bendType == "COMPLEMENT":
            s += "l"

        s += "(%d,%d,%d)" % (self.A + 1, self.B + 1, self.C + 1)
        if self.fixedEqVal:
            s += "[%.1f]" % (self.fixedEqVal * self.qShowFactor)
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms: return False
        elif not isinstance(other, BEND): return False
        elif self.bendType != other.bendType: return False
        else: return True

    @property
    def bendType(self):
        return self._bendType

    @bendType.setter
    def bendType(self, intype):
        if intype in "REGULAR" "LINEAR" "COMPLEMENT":
            self._bendType = intype
        else:
            raise optExceptions.OPT_FAIL(
                "BEND.bendType must be REGULAR, LINEAR, or COMPLEMENT")

    def compute_axes(self, geom):
        check, u = v3d.eAB(geom[self.B], geom[self.A])  # B->A
        check, v = v3d.eAB(geom[self.B], geom[self.C])  # B->C

        if self._bendType == "REGULAR":  # not a linear-bend type
            self._w[:] = v3d.cross(u, v)  # orthogonal vector
            v3d.normalize(self._w)
            self._x[:] = u + v  # angle bisector
            v3d.normalize(self._x)
            return

        tv1 = np.array([1, 0, 0], float)  # hope not to create 2 bends that both break
        tv2 = np.array([0, 1, 1], float)  # a symmetry plane, so 2nd is off-axis
        v3d.normalize(tv2)

        # handle both types of linear bends
        if not v3d.are_parallel_or_antiparallel(u, v):
            self._w[:] = v3d.cross(u, v)  # orthogonal vector
            v3d.normalize(self._w)
            self._x[:] = u + v  # angle bisector
            v3d.normalize(self._x)

        # u || v but not || to tv1.
        elif not v3d.are_parallel_or_antiparallel(u,tv1)  \
         and not v3d.are_parallel_or_antiparallel(v,tv1):
            self._w[:] = v3d.cross(u, tv1)
            v3d.normalize(self._w)
            self._x[:] = v3d.cross(self._w, u)
            v3d.normalize(self._x)

        # u || v but not || to tv2.
        elif not v3d.are_parallel_or_antiparallel(u,tv2) \
         and not v3d.are_parallel_or_antiparallel(v,tv2):
            self._w[:] = v3d.cross(u, tv2)
            v3d.normalize(self._w)
            self._x[:] = v3d.cross(self._w, u)
            v3d.normalize(self._x)

        if self._bendType == "COMPLEMENT":
            w2 = np.copy(self._w)  # x_normal -> w_complement
            self._w[:] = -1.0 * self._x  # -w_normal -> x_complement
            self._x[:] = w2
            del w2

        return

    def q(self, geom):
        #check, phi = v3d.angle(geom[self.A], geom[self.B], geom[self.C])
        #print('Traditional Angle = %15.10f\n', phi)

        if not self._axes_fixed:
            self.compute_axes(geom)

        check, u = v3d.eAB(geom[self.B], geom[self.A])  # B->A
        check, v = v3d.eAB(geom[self.B], geom[self.C])  # B->C

        # linear bend is sum of 2 angles, u.x + v.x
        origin = np.zeros(3, float)
        check, phi = v3d.angle(u, origin, self._x)
        if not check:
            raise optExceptions.ALG_FAIL("BEND.q could not compute linear bend")

        check, phi2 = v3d.angle(self._x, origin, v)
        if not check:
            raise optExceptios.ALG_FAIL("BEND.q could not compute linear bend")
        phi += phi2
        return phi

    @property
    def qShowFactor(self):
        return 180.0 / np.pi

    def qShow(self, geom):  # return in degrees
        return self.q(geom) * self.qShowFactor

    @property
    def fShowFactor(self):
        return HARTREE2AJ * np.pi / 180.0

    @staticmethod
    def zeta(a, m, n):
        if a == m: return 1
        elif a == n: return -1
        else: return 0

    def fixBendAxes(self, geom):
        if self.bendType == 'LINEAR' or self.bendType == 'COMPLEMENT':
            self.compute_axes(geom)
            self._axes_fixed = True

    def unfixBendAxes(self):
        self._axes_fixed = False

    def DqDx(self, geom, dqdx):
        if not self._axes_fixed:
            self.compute_axes(geom)

        u = geom[self.A] - geom[self.B]  # B->A
        v = geom[self.C] - geom[self.B]  # B->C
        Lu = v3d.norm(u)  # RBA
        Lv = v3d.norm(v)  # RBC
        u[:] *= 1.0 / Lu  # u = eBA
        v[:] *= 1.0 / Lv  # v = eBC

        uXw = v3d.cross(u, self._w)
        wXv = v3d.cross(self._w, v)

        # B = overall index of atom; a = 0,1,2 relative index for delta's
        for a, B in enumerate(self.atoms):
            dqdx[3*B : 3*B+3] = BEND.zeta(a,0,1) * uXw[0:3]/Lu + \
                                BEND.zeta(a,2,1) * wXv[0:3]/Lv
        return

    # Return derivative B matrix elements.  Matrix is cart X cart and passed in.
    def Dq2Dx2(self, geom, dq2dx2):

        if not self._axes_fixed:
            self.compute_axes(geom)

        u = geom[self.A] - geom[self.B]  # B->A
        v = geom[self.C] - geom[self.B]  # B->C
        Lu = v3d.norm(u)  # RBA
        Lv = v3d.norm(v)  # RBC
        u *= 1.0 / Lu  # eBA
        v *= 1.0 / Lv  # eBC

        uXw = v3d.cross(u, self._w)
        wXv = v3d.cross(self._w, v)

        # packed, or mini dqdx where columns run only over 3 atoms
        dqdx = np.zeros(9, float)
        for a in range(3):
            dqdx[3*a : 3*a+3] = BEND.zeta(a,0,1) * uXw[0:3]/Lu + \
                                BEND.zeta(a,2,1) * wXv[0:3]/Lv

        val = self.q(geom)
        cos_q = cos(val)  # cos_q = v3d_dot(u,v);

        if 1.0 - cos_q * cos_q <= 1.0e-12:  # leave 2nd derivatives empty - sin 0 = 0 in denominator
            return
        sin_q = sqrt(1.0 - cos_q * cos_q)

        for a in range(3):
            for i in range(3):  #i = a_xyz
                for b in range(3):
                    for j in range(3):  #j=b_xyz
                        tval =  BEND.zeta(a,0,1) * BEND.zeta(b,0,1) * \
                          (u[i]*v[j]+u[j]*v[i]-3*u[i]*u[j]*cos_q+delta(i,j)*cos_q) / (Lu*Lu*sin_q)

                        tval += BEND.zeta(a,2,1) * BEND.zeta(b,2,1) * \
                          (v[i]*u[j]+v[j]*u[i]-3*v[i]*v[j]*cos_q+delta(i,j)*cos_q) / (Lv*Lv*sin_q)

                        tval += BEND.zeta(a,0,1) * BEND.zeta(b,2,1) * \
                          (u[i]*u[j]+v[j]*v[i]-u[i]*v[j]*cos_q-delta(i,j)) / (Lu*Lv*sin_q)

                        tval += BEND.zeta(a,2,1) * BEND.zeta(b,0,1) * \
                          (v[i]*v[j]+u[j]*u[i]-v[i]*u[j]*cos_q-delta(i,j)) / (Lu*Lv*sin_q)

                        tval -= cos_q / sin_q * dqdx[3 * a + i] * dqdx[3 * b + j]

                        dq2dx2[3 * self.atoms[a] + i, 3 * self.atoms[b] + j] = tval

        return

    def diagonalHessianGuess(self, geom, Z, connectivity=False, guessType="SIMPLE"):
        """ Generates diagonal empirical Hessians in a.u. such as 
          Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
          Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """
        if guessType == "SIMPLE":
            return 0.2

        elif guessType == "SCHLEGEL":
            if Z[self.A] == 1 or Z[self.C] == 1:
                return 0.160
            else:
                return 0.250

        elif guessType == "FISCHER":
            a = 0.089
            b = 0.11
            c = 0.44
            d = -0.42
            Rcov_AB = (
                covRadii.R[int(Z[self.A])] + covRadii.R[int(Z[self.B])]) / BOHR2ANGSTROMS
            Rcov_BC = (
                covRadii.R[int(Z[self.C])] + covRadii.R[int(Z[self.B])]) / BOHR2ANGSTROMS
            R_AB = v3d.dist(geom[self.A], geom[self.B])
            R_BC = v3d.dist(geom[self.B], geom[self.C])
            return a + b / (np.power(Rcov_AB * Rcov_BC, d)) * np.exp(
                -c * (R_AB + R_BC - Rcov_AB - Rcov_BC))

        elif guessType == "LINDH_SIMPLE":
            R_AB = v3d.dist(geom[self.A], geom[self.B])
            R_BC = v3d.dist(geom[self.B], geom[self.C])
            k_phi = 0.15
            Lindh_Rho_AB = HguessLindhRho(Z[self.A], Z[self.B], R_AB)
            Lindh_Rho_BC = HguessLindhRho(Z[self.B], Z[self.C], R_BC)
            return k_phi * Lindh_Rho_AB * Lindh_Rho_BC

        else:
            print("Warning: Hessian guess encountered unknown coordinate type.\n")
            return 1.0
