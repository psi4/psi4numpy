from abc import ABCMeta, abstractmethod
from . import optExceptions


class SIMPLE(object):
    __metaclass__ = ABCMeta

    def __init__(self, atoms, frozen=False, fixedEqVal=None):
        # these lines use the property's and setters below
        self.atoms = atoms  # atom indices for internal definition
        self.frozen = frozen  # bool - is internal coordinate frozen?
        self.fixedEqVal = fixedEqVal  # target value if artificial forces are to be added

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, values):
        try:
            for v in values:
                if int(v) < 0:
                    raise optExceptions.OPT_FAIL('Atom identifier cannot be negative.')
        except:
            raise optExceptions.OPT_FAIL('Atoms must be iterable list of whole numbers.')
        self._atoms = values

    @property
    def frozen(self):
        return self._frozen

    @property
    def fixed(self):
        if self._fixedEqVal is None:
            return False
        else:
            return True

    @frozen.setter
    def frozen(self, setval):
        self._frozen = bool(setval)
        return

    @property
    def fixedEqVal(self):
        return self._fixedEqVal

    @fixedEqVal.setter
    def fixedEqVal(self, qTarget=None):
        if qTarget is not None:
            try:
                float(qTarget)
            except:
                raise optExceptions.OPT_FAIL("Eq. value must be a float or None.")
        self._fixedEqVal = qTarget

    @property
    def A(self):
        try:
            return self.atoms[0]
        except:
            raise optExceptions.OPT_FAIL("A() called but atoms[0] does not exist")

    @property
    def B(self):
        try:
            return self.atoms[1]
        except:
            raise optExceptions.OPT_FAIL("B() called but atoms[1] does not exist")

    @property
    def C(self):
        try:
            return self.atoms[2]
        except:
            raise optExceptions.OPT_FAIL("C() called but atoms[2] does not exist")

    @property
    def D(self):
        try:
            return self.atoms[3]
        except:
            raise optExceptions.OPT_FAIL("D() called but atoms[3] does not exist")

    # ** constructor + 7 abstract methods are currently required **
    @abstractmethod  # Given geometry, return value in Bohr or radians
    def q(self, geom):
        pass

    @abstractmethod  # Given geometry, return Value in Angstroms or degrees.
    def qShow(self, geom):
        pass

    @abstractmethod  # Return the scalar needed to convert value in au to Ang or Deg
    def qShowFactor(self):
        pass

    @abstractmethod  # Return the scalar needed to convert force in au to aJ/(Ang or Deg)
    def fShowFactor(self):
        pass

    # Modify provided DqDx array with first derivative of value wrt cartesians
    #  i.e., provide rows of B matrix.
    #   Num. of rows is len(self._atoms), or Num. of atoms in coordinate definition
    # By default, col dimension of dqdx is assumed to be 3*(Num. of atoms in fragment,
    #  or the number of atoms consistent with the values of self._atoms).
    # If mini==True, then col dimension of dqdx is only 3*len(self._atoms).  For a stretch
    # then, e.g, DqDx is 2x6.
    @abstractmethod
    def DqDx(self, geom, dqdx, mini=False):
        raise optExceptions.ALG_FAIL('no DqDx for this coordinate')

    # Modify provided Dq2Dx2 array with second derivative of value wrt cartesians
    #  i.e., provide derivative B matrix for coordinate.
    # dimension of dq2dx2 is 3*len(self._atoms)x3*len(self._atoms), or
    # cartesian by cartesian - of minimum size.
    @abstractmethod  # Derivative of value wrt cartesians, i.e., B-matrix elements.
    def Dq2Dx2(self, geom, dq2dx2):
        raise optExceptions.ALG_FAIL('no Dq2Dx2 for this coordinate')

    @abstractmethod  # Diagonal hessian guess
    def diagonalHessianGuess(geom, Z, connectivity, guessType):
        raise optExceptions.ALG_FAIL('no hessian guess for this coordinate')
