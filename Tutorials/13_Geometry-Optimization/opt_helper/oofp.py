from math import cos, sin, tan

from . import optExceptions
from . import v3d
from .simple import *

FIX_VAL_NEAR_PI = 1.57
from psi4 import constants
HARTREE2AJ = constants.hartree2aJ

# Class for out-of-plane angle.  Definition (A,B,C,D) means angle AB with respect
# to the CBD plane; canonical order is C < D


class OOFP(SIMPLE):
    def __init__(self, a, b, c, d, frozen=False, fixedEqVal=None):

        if c < d: atoms = (a, b, c, d)
        else: atoms = (a, b, d, c)
        self._near180 = 0
        SIMPLE.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen: s = '*'
        else: s = ' '

        s += "O"

        s += "(%d,%d,%d,%d)" % (self.A + 1, self.B + 1, self.C + 1, self.D + 1)
        if self.fixedEqVal:
            s += "[%.4f]" % self.fixedEqVal
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms: return False
        elif not isinstance(other, OOFP): return False
        else: return True

    @property
    def near180(self):
        return self._near180

    def updateOrientation(self, geom):
        tval = self.q(geom)
        if tval > FIX_VAL_NEAR_PI:
            self._near180 = +1
        elif tval < -1 * FIX_VAL_NEAR_PI:
            self._near180 = -1
        else:
            self._near180 = 0
        return

    @property
    def qShowFactor(self):
        return 180.0 / np.pi

    def qShow(self, geom):  # return in degrees
        return self.q(geom) * self.qShowFactor

    @property
    def fShowFactor(self):
        return HARTREE2AJ * np.pi / 180.0

    # compute angle and return value in radians
    def q(self, geom):
        check, tau = v3d.oofp(geom[self.A], geom[self.B], geom[self.C], geom[self.D])
        if not check:
            raise optExceptions.ALG_FAIL(
                "OOFP::compute.q: unable to compute out-of-plane value")

        # Extend domain of out-of-plane angles to beyond pi
        if self._near180 == -1 and tau > FIX_VAL_NEAR_PI:
            return tau - 2.0 * np.pi
        elif self._near180 == +1 and tau < -1 * FIX_VAL_NEAR_PI:
            return tau + 2.0 * np.pi
        else:
            return tau

    # out-of-plane is m-o-p-n
    # Assume angle phi_CBD is OK, or we couldn't calculate the value anyway.
    def DqDx(self, geom, dqdx):
        eBA = geom[self.A] - geom[self.B]
        eBC = geom[self.C] - geom[self.B]
        eBD = geom[self.D] - geom[self.B]
        rBA = v3d.norm(eBA)
        rBC = v3d.norm(eBC)
        rBD = v3d.norm(eBD)
        eBA *= 1.0 / rBA
        eBC *= 1.0 / rBC
        eBD *= 1.0 / rBD

        # compute out-of-plane value, C-B-D angle
        val = self.q(geom)
        check, phi_CBD = v3d.angle(geom[self.C], geom[self.B], geom[self.D])

        # S vector for A
        tmp = v3d.cross(eBC, eBD)
        tmp /= cos(val) * sin(phi_CBD)
        tmp2 = tan(val) * eBA
        dqdx[3 * self.A:3 * self.A + 3] = (tmp - tmp2) / rBA

        # S vector for C
        tmp = v3d.cross(eBD, eBA)
        tmp = tmp / (cos(val) * sin(phi_CBD))
        tmp2 = cos(phi_CBD) * eBD
        tmp3 = -1.0 * tmp2 + eBC
        tmp3 *= tan(val) / (sin(phi_CBD) * sin(phi_CBD))
        dqdx[3 * self.C:3 * self.C + 3] = (tmp - tmp3) / rBC

        # S vector for D
        tmp = v3d.cross(eBA, eBC)
        tmp /= cos(val) * sin(phi_CBD)
        tmp2 = cos(phi_CBD) * eBC
        tmp3 = -1.0 * tmp2 + eBD
        tmp3 *= tan(val) / (sin(phi_CBD) * sin(phi_CBD))
        dqdx[3 * self.D:3 * self.D + 3] = (tmp - tmp3) / rBD

        # S vector for B
        dqdx[3*self.B:3*self.B+3] = -1.0 * dqdx[3*self.A:3*self.A+3] \
           - dqdx[3*self.C:3*self.C+3] - dqdx[3*self.D:3*self.D+3]
        return

    def Dq2Dx2(self, geom, dqdx):
        raise optExceptions.ALG_FAIL('no derivative B matrices for out-of-plane angles')

    def diagonalHessianGuess(self, geom, Z, guess="SIMPLE"):
        """ Generates diagonal empirical Hessians in a.u. such as 
          Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
          Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
        """
        if guess == "SIMPLE":
            return 0.1
        else:
            print("Warning: Hessian guess encountered unknown coordinate type.\n")
            return 1.0
