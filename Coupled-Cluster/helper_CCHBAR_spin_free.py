# A simple Psi4 script to compute CCSD (spin-free) from a RHF reference
# Scipy and numpy python modules are required
#
# Algorithms were taken directly from Daniel Crawford's programming website:
# http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming
# Special thanks to Lori Burns for integral help
#
# Created by: Ashutosh Kumar
# Date: 4/29/2017
# License: GPL v3.0
#

import time
import numpy as np
import psi4

# N dimensional dot
# Like a mini DPD library
def ndot(input_string, op1, op2, prefactor=None):
    """
    No checks, if you get weird errors its up to you to debug.

    ndot('abcd,cdef->abef', arr1, arr2)
    """
    inp, output_ind = input_string.split('->')
    input_left, input_right = inp.split(',')

    size_dict = {}
    for s, size in zip(input_left, op1.shape):
        size_dict[s] = size
    for s, size in zip(input_right, op2.shape):
        size_dict[s] = size

    set_left = set(input_left)
    set_right = set(input_right)
    set_out = set(output_ind)

    idx_removed = (set_left | set_right) - set_out
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed

    # Tensordot axes
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s),)
        right_pos += (input_right.find(s),)
    tdot_axes = (left_pos, right_pos)

    # Get result ordering
    tdot_result = input_left + input_right
    for s in idx_removed:
        tdot_result = tdot_result.replace(s, '')

    rs = len(idx_removed)
    dim_left, dim_right, dim_removed = 1, 1, 1
    for key, size in size_dict.items():
        if key in keep_left:
            dim_left *= size
        if key in keep_right:
            dim_right *= size
        if key in idx_removed:
            dim_removed *= size

    shape_result = tuple(size_dict[x] for x in tdot_result)
    used_einsum = False

    # Matrix multiply
    # No transpose needed
    if input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed),
                          op2.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T,
                          op2.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed),
                          op2.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T,
                          op2.reshape(dim_removed, dim_right))

    # If we have to transpose vector-matrix, einsum is faster
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        new_view = np.einsum(input_string, op1, op2)
        used_einsum = True

    else:
        new_view = np.tensordot(op1, op2, axes=tdot_axes)

    # Make sure the resulting shape is correct
    if (new_view.shape != shape_result) and not used_einsum:
        if (len(shape_result) > 0):
            new_view = new_view.reshape(shape_result)
        else:
            new_view = np.squeeze(new_view)

    # In-place mult by prefactor if requested
    if prefactor is not None:
        new_view *= prefactor

    # Do final tranpose if needed
    if used_einsum:
        return new_view
    elif tdot_result == output_ind:
        return new_view
    else:
        return np.einsum(tdot_result + '->' + output_ind, new_view)


class helper_CCHBAR_SF(object):

    def __init__(self, ccsd, memory=2):

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nfzc = 0
        self.nocc = ccsd.ndocc 
        self.nvirt = ccsd.nmo - ccsd.nocc - ccsd.nfzc

        self.slice_nfzc = slice(0, self.nfzc)
        self.slice_o = slice(self.nfzc, self.nocc + self.nfzc)
        self.slice_v = slice(self.nocc + self.nfzc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'f': self.slice_nfzc, 'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}

       
        self.F = ccsd.F
        self.dia = ccsd.dia
        self.dijab = ccsd.dijab
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2
        print('\n..initialed CCHBAR in %.3f seconds.\n' % (time.time() - time_init))
    # occ orbitals i, j, k, l, m, n
    # virt orbitals a, b, c, d, e, f
    # all oribitals p, q, r, s, t, u, v

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

        def build_tau(self):
        ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1)
        ttau += tmp
        return ttau

        def build_L(self):
            tmp = self.MO.copy()
            L = 2.0 * tmp - tmp.swapaxes(2,3)
            return L
       
        def build_Hov(self):
            Hov = self.get_F('ov').copy()
            Hov += ndot('nf,mnef->me', self.t1, self.build_L())
            return Hov

        def build_Hoo(self):
            Hoo = self.get_F('oo').copy()
            Hoo += ndot('ie,me->mi', self.t1, self.get_F('ov'))
            Hoo += ndot('ne,mnie->mi', self.t1, self.build_L())
            Hoo += ndot('inef,mnef->mi', self.build_tau, self.build_L())
            return Hoo

        def build_Hvv(self):
            Hvv = self.get_F('vv').copy()
            Hvv -= ndot('me,ma->ae', self.get_F('ov'), self.t1)
            Hvv += ndot('mf,amef->ae', self.t1, self.build_L())
            Hvv -= ndot('mnfa,mnfe->ae', self.build_tau, self.build_L())
            return Hvv

	def build_Hoooo(self):
            Hoooo = self.get_MO('oooo').copy()
            Hoooo += ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'), prefactor=2.0)
            Hoooo += ndot('ijef,mnef->mnij', self.build_tau, self.get_MO('oovv'))
            return Hmnij

	def build_Hvvvv(self):
            Hvvvv = self.get_MO('vvvv').copy()
            Hvvvv -= ndot('mb,amef->abef', self.t1, self.get_MO('vovv'), prefactor=2.0)
            Hvvvv += ndot('mnab,mnef->mnij', self.build_tau, self.get_MO('oovv'))
            return Hvvvv
            
        def build_Hvovv(self):
            Hvovv = self.get_MO('vovv').copy()
            Hvovv -= ndot('na,nmef->amef', self.t1, self.get_MO('oovv'))
            return Hvovv

        def build_Hooov(self):
            Hooov = self.get_MO('ooov').copy()
            Hooov += ndot('if,nmef->mnie', self.t1, self.get_MO('oovv'))
            return Hooov

        def build_Hovvo(self):
            Hovvo = self.get_MO('ovvo').copy()
            Hovvo += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
            Hovvo -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))
            Hovvo -= ndot('njbf,mnef->mbej', self.build_tau(), self.get_MO('oovv'))
            Hovvo += ndot('njfb,mnef->mbej', self.t2, self.build_L())
            return Hovvo

        def build_Hovov(self):
            Hovov = self.get_MO('ovov').copy()
            Hovov += ndot('jf,bmef->mbje', self.t1, self.get_MO('vovv'))
            Hovov -= ndot('nb,mnje->mbje', self.t1, self.get_MO('ooov'))
            Hovov -= ndot('jnfb,nmef->mbje', self.build_tau(), self.get_MO('oovv'))
            return Hovov

        def build_Hvvvo(self):
            Hvvvo =  self.get_MO('vvvo').copy()
            Hvvvo += ndot('if,abef->abei', self.t1, self.get_MO('vvvv'))
            Hvvvo -= ndot('mb,amei->abei', self.t1, self.get_MO('vovo'))
            Hvvvo -= ndot('ma,bmie->abei', self.t1, self.get_MO('voov'))
            Hvvvo -= ndot('imfa,mbef->abei', self.build_tau(), self.get_MO('ovvv'))
            Hvvvo -= ndot('imfb,amef->abei', self.build_tau(), self.get_MO('vovv'))
            Hvvvo += ndot('mnab,mnei->abei', self.build_tau(), self.get_MO('oovo'))
            Hvvvo -= ndot('me,miab->abei', self.get_F('ov'), self.t2)
            Hvvvo += ndot('mifb,amef->abei', self.t2, self.build_L())
            tmp =    ndot('mnef,if->mnei', self.get_MO('oovv'), self.t1)
            Hvvvo += ndot('mnab,mnei->abei', self.t2, tmp)
            tmp =    ndot('mnef,ma->anef', self.get_MO('oovv'), self.t1)
            Hvvvo += ndot('infb,anef->abei', self.t2, tmp)
            tmp =    ndot('mnef,nb->mefb', self.get_MO('oovv'), self.t1)
            Hvvvo += ndot('miaf,mefb->abei', self.t2, tmp)
            tmp =    ndot('mnfe,mf->ne', self.build_L(), self.t1)
            Hvvvo += ndot('niab,ne->abei', self.t2, tmp)
            tmp =    ndot('mnfe,na->mafe', self.build_L(), self.t1)
            Hvvvo += ndot('mifb,mafe->abei', self.t2, tmp)
            tmp1 =   ndot('if,ma->imfa', self.t1, self.t1)
            tmp2 =   ndot('mnef,nb->mbef', self.get_MO('oovv'), self.t1)
            Hvvvo += ndot('imfa,mbef->abei', tmp1, tmp2)
            return Hvvvo

        def build_Hovoo(self):
            Hovoo =  self.get_MO('vvov').copy()
            Hovoo += ndot('je,mbie->mbij', self.t1, self.get_MO('ovov'))
            Hovoo += ndot('ie,bmje->mbij', self.t1, self.get_MO('voov'))
            Hovoo -= ndot('nb,mnij->mbij', self.t1, self.get_MO('oovv'))
            Hovoo -= ndot('ineb,nmje->mbij', self.build_tau(), self.get_MO('ooov'))
            Hovoo -= ndot('jneb,mnie->mbij', self.build_tau(), self.get_MO('ooov'))
            Hovoo += ndot('ijef,mbef->mbij', self.build_tau(), self.get_MO('ovvv'))
            Hovoo += ndot('me,ijeb->mbij', self.get_F('ov'), self.t2)
            Hovoo += ndot('njeb,mnie->mbij', self.t2, self.build_L())
            tmp =    ndot('mnef,jf->mnej', self.get_MO('oovv'), self.t1)
            Hovoo -= ndot('ineb,mnej->mbij', self.t2, tmp)
            tmp =    ndot('mnef,ie->mnif', self.get_MO('oovv'), self.t1)
            Hovoo -= ndot('jnfb,mnif->mbij', self.t2, tmp)
            tmp =    ndot('mnef,nb->mefb', self.get_MO('oovv'), self.t1)
            Hovoo -= ndot('ijef,mefb->mbij', self.t2, tmp)
            tmp =    ndot('mnef,njfb->mejb', self.build_L(), self.t2)
            Hovoo += ndot('mejb,ie->mbij', tmp, self.t1)
            tmp =    ndot('mnef,nf->me', self.build_L(), self.t1)
            Hovoo += ndot('me,ijeb->mbij', tmp, self.t2)
            tmp1 =   ndot('ie,jf->ijef', self.t1, self.t1)
            tmp2 =   ndot('mnef,nb->mbef', self.get_MO('oovv'), self.t1)
            Hovoo -= ndot('mbef,ijef->mbij', tmp2, tmp1)
            return Hovoo


# End CCHBAR class

if __name__ == "__main__":
    arr4 = np.random.rand(4, 4, 4, 4)
    arr2 = np.random.rand(4, 4)

    def test_ndot(string, op1, op2):
        ein_ret = np.einsum(string, op1, op2)
        ndot_ret = ndot(string, op1, op2)
        assert np.allclose(ein_ret, ndot_ret)

    test_ndot('abcd,cdef->abef', arr4, arr4)
    test_ndot('acbd,cdef->abef', arr4, arr4)
    test_ndot('acbd,cdef->abfe', arr4, arr4)
    test_ndot('mnab,mnij->ijab', arr4, arr4)

    test_ndot('cd,cdef->ef', arr2, arr4)
    test_ndot('ce,cdef->df', arr2, arr4)
    test_ndot('nf,naif->ia', arr2, arr4)
