# A simple Psi4 script to compute coupled cluster lambda amplitudes
# from a RHF reference
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


class helper_CCLINRESP(object):

    def __init__(self, cclambda, ccpert_x, ccpert_y):

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.pert_x = ccpert_x.pert
        self.pert_y = ccpert_y.pert

        self.l1 = cclambda.l1
        self.l2 = cclambda.l2
 
        self.x1_x = ccpert_x.x1
        self.x2_x = ccpert_x.x2
        self.y1_x = ccpert_x.y1
        self.y2_x = ccpert_x.y2
 
        self.x1_y = ccpert_y.x1
        self.x2_y = ccpert_y.x2
        self.y1_y = ccpert_y.y1
        self.y2_y = ccpert_y.y2

        print('\n..initialed CCLINRESP in %.3f seconds.\n' % (time.time() - time_init))

    def get_pert(self, pert, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return pert[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def linresp(self):
        polar1 = 0
        polar2 = 0
        polar1 += ndot("ai,ia->", ccpert_x.build_Avo(), self.y1_y)
        polar1 += ndot("abij,ijab->", ccpert_x.build_Avvoo(), self.y2_y, prefactor=0.5)
        polar1 += ndot("baji,ijab->", ccpert_x.build_Avvoo(), self.y2_y, prefactor=0.5)

        tmp = ndot('ia,jb->ijab', self.l1, self.get_pert(self.pert_x, 'ov'))
        polar2 += ndot('ijab,ijab->', tmp, self.x2_y, prefactor=2.0) 
        polar2 += ndot('ijab,ijba->', tmp, self.x2_y, prefactor=-1.0) 
 
        tmp = ndot('ia,ic->ac', self.l1, self.x1_y)
        polar2 += ndot('ac,ac->', tmp, ccpert_x.build_Avv()) 
        tmp = ndot('ia,ka->ik', self.l1, self.x1_y)
        polar2 -= ndot('ik,ki->', tmp, ccpert_x.build_Aoo()) 

        tmp = ndot('ijbc,bcaj->ia', self.l2, ccpert_x.build_Avvvo())
        polar2 += ndot('ia,ia->', tmp, self.x1_y)) 

        tmp = ndot('ijab,kbij->ak', self.l2, ccpert_x.build_Aovoo())
        polar2 -= ndot('ak,ka->', tmp, self.x1_y), prefactor=0.5) 

        tmp = ndot('ijab,kaji->bk', self.l2, ccpert_x.build_Aovoo())
        polar2 -= ndot('bk,kb->', tmp, self.x1_y), prefactor=0.5) 

        tmp = ndot('ijab,kjab->ik', self.l2, self.x2_y)
        polar2 -= ndot('ik,ki->', tmp, ccpert_x.build_Aoo()), prefactor=0.5) 

        tmp = ndot('ijab,kiba->jk', self.l2, self.x2_y,)
        polar2 -= ndot('jk,kj->', tmp, ccpert_x.build_Aoo()), prefactor=0.5) 

        tmp = ndot('ijab,ijac->bc', self.l2, self.x2_y,)
        polar2 += ndot('bc,bc->', tmp, ccpert_x.build_Avv()), prefactor=0.5) 

        tmp = ndot('ijab,ijcb->ac', self.l2, self.x2_y,)
        polar2 += ndot('ac,ac->', tmp, ccpert_x.build_Avv()), prefactor=0.5) 
      
        polar2 += ndot("ia,ia->", self.get_pert(self.pert_x, 'ov'), self.x1_y)

        return -1.0 * (polar1 + polar2)

# End CCLINRESP class

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
