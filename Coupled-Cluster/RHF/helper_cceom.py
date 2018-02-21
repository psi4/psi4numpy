from utils import ndot


class HelperCCEom(object):
    """
    EOMCCSD helper class for spin adapted EOMCCSD

    """

    def __init__(self, ccsd, cchbar):
        """
        Initializes the HelperCCEom object

        Parameters
        ----------
        ccsd: HelperCCSd object
            Energy should already be computed

        cchbar: HelperCCHbar object


        Returns
        -------
        ret : HelperCCEom
            An initialized HelperCCEom object

        Notes
        -----
        Spin orbital sigma equations for EOMCCSDT can be found in:
            I. Shavitt and R. J. Bartlett, "Many-Body Methods in Chemistry and
            Physics: MBPT and Coupled-Cluster Theory", Cambridge University
            Press, 2009.
        The relevant contributions for EOMCCSD were extracted and the equations
        spin adapted to arrive at the equations implemented in this class.

        Special thanks to Ashutosh Kumar for Hbar components and help with spin
        adaptation.

        """
        # Steal dimensions
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nocc = ccsd.ndocc
        self.nvir = ccsd.nmo - ccsd.nocc
        self.nsingles = self.ndocc * self.nvir
        self.ndoubles = self.ndocc * self.ndocc * self.nvir * self.nvir

        # Steal integrals/amps from ccsd
        self.MO = ccsd.MO
        self.F = ccsd.F
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2

        # Steal "ova" translation
        self.slice_o = cchbar.slice_o
        self.slice_v = cchbar.slice_v
        self.slice_a = cchbar.slice_a
        self.slice_dict = cchbar.slice_dict

        # Steal Hbar blocks
        self.Hov = cchbar.Hov
        self.Hoo = cchbar.Hoo
        self.Hvv = cchbar.Hvv
        self.Hoooo = cchbar.Hoooo
        self.Hvvvv = cchbar.Hvvvv
        self.Hvovv = cchbar.Hvovv
        self.Hooov = cchbar.Hooov
        self.Hovvo = cchbar.Hovvo
        self.Hovov = cchbar.Hovov
        self.Hvvvo = cchbar.Hvvvo
        self.Hovoo = cchbar.Hovoo
        self.Loovv = cchbar.Loovv

        # Build Approximate Diagonal of Hbar
        self.Dia = self.Hoo.diagonal().reshape(-1, 1) - self.Hvv.diagonal()
        self.Dijab = self.Hoo.diagonal().reshape(
            -1, 1, 1, 1) + self.Hoo.diagonal().reshape(
                -1, 1, 1) - self.Hvv.diagonal().reshape(
                    -1, 1) - self.Hvv.diagonal()

    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def build_sigma1(self, B1, B2):
        """
        Compute the contributions to <ia|Hbar*B|0>

        Parameters
        ----------
        B1: array like, shape(ndocc, nvir)
          The first nsingles elements of a guess vector reshaped to (o,v)

        B2: array like, shape(ndocc,ndocc,nvir,nvir)
          The last ndoubles elements of a guess vector reshaped to (o,o,v,v)

        Returns
        -------
        S1: ndarray shape(ndocc, nvir)

        Examples
        --------

        >>> # Get some vectors as cols of a 2D numpy array and orthogonalize them
        >>> c  = np.random.rand(eom.nsingles + eom.ndoubles, 2)
        >>> c,  = np.linalg.qr(c)

        >>> # Slice out the singles, doubles blocks of the first vector and reshape
        >>> B1 = c[:,:nsingles].reshape(eom.ndocc, eom.nvir)
        >>> B2 = c[:,nsingles:].reshape(eom.ndocc, eom.ndocc, eom.nvir, eom.nvir)
        >>> S1 = eom.build_sigma1(B1, B2)

        """
        S1 = ndot('ie,ae->ia', B1, self.Hvv)
        S1 -= ndot('mi,ma->ia', self.Hoo, B1)
        S1 += ndot('maei,me->ia', self.Hovvo, B1, prefactor=2.0)
        S1 += ndot('maie,me->ia', self.Hovov, B1, prefactor=-1.0)
        S1 += ndot('miea,me->ia', B2, self.Hov, prefactor=2.0)
        S1 += ndot('imea,me->ia', B2, self.Hov, prefactor=-1.0)
        S1 += ndot('imef,amef->ia', B2, self.Hvovv, prefactor=2.0)
        S1 += ndot('imef,amfe->ia', B2, self.Hvovv, prefactor=-1.0)
        S1 -= ndot('mnie,mnae->ia', self.Hooov, B2, prefactor=2.0)
        S1 -= ndot('nmie,mnae->ia', self.Hooov, B2, prefactor=-1.0)
        return S1

    def build_sigma2(self, B1, B2):
        """
        Compute the contributions to <ijab|Hbar*B|0>:

        Parameters
        ----------
        B1: array like, shape(ndocc, nvir)
          The first nsingles elements of a guess vector reshaped to (o,v)

        B2: array like, shape(ndocc,ndocc,nvir,nvir)
          The last ndoubles elements of a guess vector reshaped to (o,o,v,v)

        Returns
        -------
        S2: ndarray shape(ndocc, ndocc, nvir, nvir)

        Examples
        --------

        >>> # Get some vectors as cols of a 2D numpy array and orthogonalize them
        >>> c  = np.random.rand(eom.nsingles + eom.ndoubles, 2)
        >>> c,  = np.linalg.qr(c)

        >>> # Slice out the singles, doubles blocks of the first vector and reshape
        >>> B1 = c[:,:nsingles].reshape(eom.ndocc, eom.nvir)
        >>> B2 = c[:,nsingles:].reshape(eom.ndocc, eom.ndocc, eom.nvir, eom.nvir)
        >>> S2 = eom.build_sigma2(B1, B2)

        """
        S_2 = ndot('ie,abej->ijab', B1, self.Hvvvo)
        S_2 -= ndot('mbij,ma->ijab', self.Hovoo, B1)

        Zvv = ndot("amef,mf->ae", self.Hvovv, B1, prefactor=2.0)
        Zvv += ndot("amfe,mf->ae", self.Hvovv, B1, prefactor=-1.0)
        Zvv -= ndot('nmaf,nmef->ae', B2, self.Loovv)
        S_2 += ndot('ijeb,ae->ijab', self.t2, Zvv)

        Zoo = ndot('mnie,ne->mi', self.Hooov, B1, prefactor=-2.0)
        Zoo -= ndot('nmie,ne->mi', self.Hooov, B1, prefactor=-1.0)
        Zoo -= ndot('mnef,inef->mi', self.Loovv, B2)
        S_2 += ndot('mi,mjab->ijab', Zoo, self.t2)

        S_2 += ndot('ijeb,ae->ijab', B2, self.Hvv)
        S_2 -= ndot('mi,mjab->ijab', self.Hoo, B2)

        S_2 += ndot('mnij,mnab->ijab', self.Hoooo, B2, prefactor=0.5)
        S_2 += ndot('ijef,abef->ijab', B2, self.Hvvvv, prefactor=0.5)

        S_2 -= ndot('imeb,maje->ijab', B2, self.Hovov)
        S_2 -= ndot('imea,mbej->ijab', B2, self.Hovvo)

        S_2 += ndot('miea,mbej->ijab', B2, self.Hovvo, prefactor=2.0)
        S_2 += ndot('miea,mbje->ijab', B2, self.Hovov, prefactor=-1.0)
        return S_2 + S_2.swapaxes(0, 1).swapaxes(2, 3)
