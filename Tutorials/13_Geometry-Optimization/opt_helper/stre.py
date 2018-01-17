import numpy as np

from . import covRadii
from . import optExceptions
from . import v3d
from .misc import delta, ZtoPeriod, HguessLindhRho
from .simple import *

from psi4 import constants
BOHR2ANGSTROMS = constants.bohr2angstroms
HARTREE2AJ = constants.hartree2aJ

class STRE(SIMPLE):
    def __init__(self, a, b, frozen=False, fixedEqVal=None, inverse=False):

        self._inverse = inverse  # bool - is really 1/R coordinate?

        if a < b: atoms = (a, b)
        else: atoms = (b, a)

        SIMPLE.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen: s = '*'
        else: s = ' '

        if self.inverse: s += '1/R'
        else: s += 'R'

        s += "(%d,%d)" % (self.A + 1, self.B + 1)
        if self.fixedEqVal:
            s += "[%.4f]" % (self.fixedEqVal * self.qShowFactor)
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms: return False
        elif not isinstance(other, STRE): return False
        elif self.inverse != other.inverse: return False
        else: return True

    @property
    def inverse(self):
        return self._inverse

    @inverse.setter
    def inverse(self, setval):
        self._inverse = bool(setval)

    def q(self, geom):
        return v3d.dist(geom[self.A], geom[self.B])

    def qShow(self, geom):
        return self.qShowFactor * self.q(geom)

    @property
    def qShowFactor(self):
        return BOHR2ANGSTROMS

    @property
    def fShowFactor(self):
        return HARTREE2AJ / BOHR2ANGSTROMS

    # If mini == False, dqdx is 1x(3*number of atoms in fragment).
    # if mini == True, dqdx is 1x6.
    def DqDx(self, geom, dqdx, mini=False):
        check, eAB = v3d.eAB(geom[self.A], geom[self.B])  # A->B
        if not check:
            raise optExceptions.ALG_FAIL("STRE.DqDx: could not normalize s vector")

        if mini:
            startA = 0
            startB = 3
        else:
            startA = 3 * self.A
            startB = 3 * self.B

        dqdx[startA:startA + 3] = -1 * eAB[0:3]
        dqdx[startB:startB + 3] = eAB[0:3]

        if self._inverse:
            val = self.q(geom)
            dqdx[startA:startA + 3] *= -1.0 * val * val  # -(1/R)^2 * (dR/da)
            dqdx[startB:startB + 3] *= -1.0 * val * val

        return

    # Return derivative B matrix elements.  Matrix is cart X cart and passed in.
    def Dq2Dx2(self, geom, dq2dx2):
        check, eAB = v3d.eAB(geom[self.A], geom[self.B])  # A->B
        if not check:
            raise optExceptions.ALG_FAIL("STRE.Dq2Dx2: could not normalize s vector")

        if not self._inverse:
            length = self.q(geom)

            for a in range(2):
                for a_xyz in range(3):
                    for b in range(2):
                        for b_xyz in range(3):
                            tval = (
                                eAB[a_xyz] * eAB[b_xyz] - delta(a_xyz, b_xyz)) / length
                            if a == b:
                                tval *= -1.0
                            dq2dx2[3*self.atoms[a]+a_xyz, \
                                3*self.atoms[b]+b_xyz] = tval

        else:  # using 1/R
            val = self.q(geom)

            dqdx = np.zeros((3 * len(self.atoms)), float)
            self.DqDx(geom, dqdx, mini=True)  # returned matrix is 1x6 for stre

            for a in range(a):
                for a_xyz in range(3):
                    for b in range(b):
                        for b_xyz in range(3):
                            dq2dx2[3*self.atoms[a]+a_xyz, 3*self.atoms[b]+b_xyz] \
                                = 2.0 / val * dqdx[3*a+a_xyz] * dqdx[3*b+b_xyz]

        return

    def diagonalHessianGuess(self, geom, Z, connectivity=False, guessType="SIMPLE"):
        """ Generates diagonal empirical Hessians in a.u. such as
		  Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
		  Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
		"""
        if guessType == "SIMPLE":
            return 0.5

        if guessType == "SCHLEGEL":
            R = v3d.dist(geom[self.A], geom[self.B])
            PerA = ZtoPeriod(Z[self.A])
            PerB = ZtoPeriod(Z[self.B])

            AA = 1.734
            if PerA == 1:
                if PerB == 1:
                    BB = -0.244
                elif PerB == 2:
                    BB = 0.352
                else:
                    BB = 0.660
            elif PerA == 2:
                if PerB == 1:
                    BB = 0.352
                elif PerB == 2:
                    BB = 1.085
                else:
                    BB = 1.522
            else:
                if PerB == 1:
                    BB = 0.660
                elif PerB == 2:
                    BB = 1.522
                else:
                    BB = 2.068

            F = AA / ((R - BB) * (R - BB) * (R - BB))
            return F

        elif guessType == "FISCHER":
            Rcov = (
                covRadii.R[int(Z[self.A])] + covRadii.R[int(Z[self.B])]) / BOHR2ANGSTROMS
            R = v3d.dist(geom[self.A], geom[self.B])
            AA = 0.3601
            BB = 1.944
            return AA * (np.exp(-BB * (R - Rcov)))

        elif guessType == "LINDH_SIMPLE":
            R = v3d.dist(geom[self.A], geom[self.B])
            k_r = 0.45
            return k_r * HguessLindhRho(Z[self.A], Z[self.B], R)

        else:
            print("Warning: Hessian guess encountered unknown coordinate type.\n")
            return 1.0


class HBOND(STRE):
    def __str__(self):
        if self.frozen: s = '*'
        else: s = ' '

        if self.inverse: s += '1/H'
        else: s += 'H'

        s += "(%d,%d)" % (self.A + 1, self.B + 1)
        if self.fixedEqVal:
            s += "[%.4f]" % self.fixedEqVal
        return s

    # overrides STRE eq in comparisons, regardless of order
    def __eq__(self, other):
        if self.atoms != other.atoms: return False
        elif not isinstance(other, HBOND): return False
        elif self.inverse != other.inverse: return False
        else: return True

    def diagonalHessianGuess(self, geom, Z, connectivity, guessType):
        """ Generates diagonal empirical Hessians in a.u. such as
		  Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
		  Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
		"""
        if guess == "SIMPLE":
            return 0.1
        else:
            print("Warning: Hessian guess encountered unknown coordinate type.\n")
            return 1.0
