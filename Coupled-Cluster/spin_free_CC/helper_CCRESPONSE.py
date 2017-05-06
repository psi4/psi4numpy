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


class helper_CCRESPONSE(object):

    def __init__(self, pert, ccsd, hbar, cclambda, memory=2):

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.pert = pert
        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nfzc = 0
        self.nocc = ccsd.ndocc 
        self.nvirt = ccsd.nmo - ccsd.nocc - ccsd.nfzc

        self.mints = ccsd.mints

        self.slice_nfzc = slice(0, self.nfzc)
        self.slice_o = slice(self.nfzc, self.nocc + self.nfzc)
        self.slice_v = slice(self.nocc + self.nfzc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'f': self.slice_nfzc, 'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}

       
        self.F = ccsd.F
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2


        self.ttau  =  hbar.ttau
        self.L     =  hbar.L
        self.Hov   =  hbar.Hov
        self.Hvv   =  hbar.Hvv
        self.Hoo   =  hbar.Hoo
        self.Hoooo =  hbar.Hoooo
        self.Hvvvv =  hbar.Hvvvv
        self.Hvovv =  hbar.Hvovv
        self.Hooov =  hbar.Hooov
        self.Hovvo =  hbar.Hovvo
        self.Hovov =  hbar.Hovov
        self.Hvvvo =  hbar.Hvvvo
        self.Hovoo =  hbar.Hovoo


        self.l1 = cclambda.l1
        self.l2 = cclambda.l2

        self.omega = 0
        self.Dia = self.Hoo.diagonal().reshape(-1, 1) - self.Hvv.diagonal()        
        self.Dijab = self.Hoo.diagonal().reshape(-1, 1, 1, 1) + self.Hoo.diagonal().reshape(-1, 1, 1) - self.Hvv.diagonal().reshape(-1, 1) - self.Hvv.diagonal() + self.omega
 
        self.x1 = self.build_Avo().swapaxes(0,1)/(self.Dia + self.omega)
        self.y1 = self.build_Avo().swapaxes(0,1)/(self.Dia + self.omega)
       
        tmp = self.build_Avvoo()
        tmp += tmp.swapaxes(0,1).swapaxes(2,3)  
        self.x2 = tmp.swapaxes(0,2).swapaxes(1,3)/(self.Dijab + self.omega)
        self.y2 = tmp.swapaxes(0,2).swapaxes(1,3)/(self.Dijab + self.omega)
 
        print('\n..initialed CCRESPONSE in %.3f seconds.\n' % (time.time() - time_init))
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

    def get_L(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return (self.L[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]])


    def get_pert(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.pert[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def build_Aoo(self):
        Aoo = self.get_pert('oo').copy()
        Aoo += ndot('ie,me->mi', self.t1, self.get_pert('ov'))
        return Aoo

    def build_Aov(self):
        Aov = self.get_pert('ov').copy()
        return Aov

    def build_Avo(self):
        Avo =  self.get_pert('vo').copy()
        Avo += ndot('ae,ie->ai', self.get_pert('vv'), self.t1)
        Avo -= ndot('ma,mi->ai', self.t1, self.get_pert('oo'))
        Avo += ndot('miea,me->ai', self.t2, self.get_pert('ov'), prefactor=2.0)
        Avo += ndot('imea,me->ai', self.t2, self.get_pert('ov'), prefactor=-1.0)
        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        Avo -= ndot('imea,me->ai', tmp, self.get_pert('ov'))
        return Avo 

    def build_Avv(self):
        Avv =  self.get_pert('vv').copy()
        Avv -= ndot('ma,me->ae', self.t1, self.get_pert('ov'))
        return Avv
        
    def build_Aovoo(self):
        Aovoo = 0
        Aovoo += ndot('ijeb,me->mbij', self.t2, self.get_pert('ov'))
        return Aovoo

    def build_Avvvo(self):
        Avvvo = 0
        Avvvo -= ndot('miab,me->abei', self.t2, self.get_pert('ov'))
        return Avvvo

    def build_Avvoo(self):
        Avvoo = 0
        Avvoo += ndot('ijeb,ae->abij', self.t2, self.build_Avv())
        Avvoo -= ndot('mjab,mi->abij', self.t2, self.build_Aoo())
        return Avvoo


    def build_Goo(self, t2, y2):
        Goo = 0
        Goo += ndot('mjab,ijab->mi', t2, y2)
        return Goo

    def build_Gvv(self, y2, t2):
        Gvv = 0
        Gvv -= ndot('ijab,ijeb->ae', y2, t2)
        return Gvv

    def build_Zvv(self):
        Zvv = 0
        Zvv += ndot('amef,mf->ae', self.Hvovv, self.x1, prefactor=2.0)
        Zvv += ndot('amfe,mf->ae', self.Hvovv, self.x1, prefactor=-1.0)
        Zvv -= ndot('mnaf,mnef->ae', self.x2, self.get_L('oovv'))
        return Zvv

    def build_Zoo(self):
        Zoo = 0
        Zoo -= ndot('mnie,ne->mi', self.Hooov, self.x1, prefactor=2.0)
        Zoo -= ndot('nmie,ne->mi', self.Hooov, self.x1, prefactor=-1.0)
        Zoo -= ndot('mnef,inef->mi', self.get_L('oovv'), self.x2)
        return Zoo

    def build_x1l1oo(self, x1, l1):
        x1l1oo = 0
        x1l1oo += ndot('me,ie->mi', x1, l1)
        return x1l1oo
        
    def build_x1l1vv(self, x1, l1):
        x1l1vv = 0
        x1l1vv += ndot('me,ma->ea', x1, l1)
        return x1l1vv

    def build_x1l1ovov(self, x1, l1):
        x1l1ovov = 0
        x1l1ovov += ndot('me,na->mena', x1, l1)
        return x1l1ovov
        

    def build_Lx2ovov(self, L, x2):
        X2Lovov = 0
        X2Lovov += ndot('imae,mnef->ianf', L, x2, prefactor=2.0)
        X2Lovov += ndot('imae,mnfe->ianf', L, x2, prefactor=-1.0)
        return X2Lovov
 
    def build_x1l2ooov(self, x1, l2):
        x1l2ooov = 0
        x1l2ooov += ndot('me,nief->mnif', x1, l2)
        return x1l2ooov
        

    def build_x1l2vovv(self, x1, l2):
        x1l2vovv = 0
        x1l2vovv += ndot('me,nmaf->enaf', x1, l2)
        return x1l2vovv


    def build_l2x2ovov_1(self, l2, x2):
        l2x2ovov = 0
        l2x2ovov += ndot('imfg,mnef->igne', l2, x2)
        return l2x2ovov

    def build_l2x2ovov_2(self, l2, x2):
        l2x2ovov = 0
        l2x2ovov += ndot('mifg,mnef->igne', l2, x2)
        return l2x2ovov

    def build_l2x2ovov_3(self, l2, x2):
        l2x2ovov = 0
        l2x2ovov += ndot('nifg,mnef->igme', l2, x2)
        return l2x2ovov


    def build_l2x2vvvv(self, l2, x2):
        l2x2vvvv = 0
        l2x2vvvv += ndot('mnga,mnef->gaef', l2, x2)
        return l2x2vvvv


    def build_l2x2oooo(self, l2, x2):
        l2x2oooo = 0
        l2x2oooo += ndot('oief,mnef->oimn', l2, x2)
        return l2x2oooo

    def Hooovx1oo(self, Hooov, x1):
        Hooovx1oo = 0
        Hooovx1oo += ndot('imne,me->in', Hooov, x1, prefactor=2.0)
        Hooovx1oo += ndot('mine,me->in', Hooov, x1, prefactor=-1.0)
        return Hooovx1oo

    def Hvovvx1vv(self, Hvovv, x1):
        Hvovvx1vv = 0
        Hvovvx1vv += ndot('fmae,me->fa', Hvovv, x1, prefactor=2.0)
        Hvovvx1vv += ndot('fmea,me->fa', Hvovv, x1, prefactor=-1.0)
        return Hvovvx1vv

    def update_X(self):
        r_x1  = self.build_Avo().swapaxes(0,1).copy() 
        r_x1 -= self.omega * self.x1.copy()
        r_x1 += ndot('ie,ae->ia', self.x1, self.Hvv)
        r_x1 -= ndot('mi,ma->ia', self.Hoo, self.x1)
        r_x1 += ndot('maei,me->ia', self.Hovvo, self.x1, prefactor=2.0)
        r_x1 += ndot('maie,me->ia', self.Hovov, self.x1, prefactor=-1.0)
        r_x1 += ndot('miea,me->ia', self.x2, self.Hov, prefactor=2.0)
        r_x1 += ndot('imea,me->ia', self.x2, self.Hov, prefactor=-1.0)
        r_x1 += ndot('imef,amef->ia', self.x2, self.Hvovv, prefactor=2.0)
        r_x1 += ndot('imef,amfe->ia', self.x2, self.Hvovv, prefactor=-1.0)
        r_x1 -= ndot('mnie,mnae->ia', self.Hooov, self.x2, prefactor=2.0)
        r_x1 -= ndot('nmie,mnae->ia', self.Hooov, self.x2, prefactor=-1.0)

        self.x1 += r_x1/self.Dia

        r_x2 = self.build_Avvoo().swapaxes(0,2).swapaxes(1,3).copy()
        r_x2 -= 0.5 * self.omega * self.x2
        r_x2 += ndot('ie,abej->ijab', self.x1, self.Hvvvo)
        r_x2 -= ndot('mbij,ma->ijab', self.Hovoo, self.x1)

        r_x2 += ndot('mi,mjab->ijab', self.build_Zoo(), self.t2)
        r_x2 += ndot('ijeb,ae->ijab', self.t2, self.build_Zvv())

        r_x2 += ndot('ijeb,ae->ijab', self.x2, self.Hvv)
        r_x2 -= ndot('mi,mjab->ijab', self.Hoo, self.x2)

        r_x2 += ndot('mnij,mnab->ijab', self.Hoooo, self.x2, prefactor=0.5)
        r_x2 += ndot('ijef,abef->ijab', self.x2, self.Hvvvv, prefactor=0.5)

        r_x2 -= ndot('imeb,maje->ijab', self.x2, self.Hovov)
        r_x2 -= ndot('imea,mbej->ijab', self.x2, self.Hovvo)

        r_x2 += ndot('miea,mbej->ijab', self.x2, self.Hovvo, prefactor=2.0)
        r_x2 += ndot('miea,mbje->ijab', self.x2, self.Hovov, prefactor=-1.0)
      
        self.x2 += r_x2/self.Dijab 
        self.x2 += r_x2.swapaxes(0,1).swapaxes(2,3)/self.Dijab
        #self.x2 = self.x2/self.Dijab

    def update_Y(self):

        # Homogenous terms (exactly same as lambda1 equations)

        r_y1  = 2.0 * self.build_Aov().copy()
        r_y1 += self.omega * self.y1
        r_y1 += ndot('ie,ea->ia', self.y1, self.Hvv)
        r_y1 -= ndot('im,ma->ia', self.Hoo, self.y1)
        r_y1 += ndot('ieam,me->ia', self.Hovvo, self.y1, prefactor=2.0)
        r_y1 += ndot('iema,me->ia', self.Hovov, self.y1, prefactor=-1.0)
        r_y1 += ndot('imef,efam->ia', self.y2, self.Hvvvo)
        r_y1 -= ndot('iemn,mnae->ia', self.Hovoo, self.y2)
        r_y1 -= ndot('eifa,ef->ia', self.Hvovv, self.build_Gvv(self.y2, self.t2), prefactor=2.0)
        r_y1 -= ndot('eiaf,ef->ia', self.Hvovv, self.build_Gvv(self.y2, self.t2), prefactor=-1.0)
        r_y1 -= ndot('mina,mn->ia', self.Hooov, self.build_Goo(self.t2, self.y2), prefactor=2.0)
        r_y1 -= ndot('imna,mn->ia', self.Hooov, self.build_Goo(self.t2, self.y2), prefactor=-1.0)

	# Inhomogenous terms 


        r_y1 += ndot('imae,me->ia', self.get_L('oovv'), self.x1, prefactor=2.0)
        r_y1 -= ndot('im,ma->ia', self.build_Aoo(), self.l1)
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Avv())
        r_y1 += ndot('imfe,feam->ia', self.l2, self.build_Avvvo())
        r_y1 -= ndot('ienm,mnea->ia', self.build_Aovoo(), self.l2, prefactor=0.5)
        r_y1 -= ndot('iemn,mnae->ia', self.build_Aovoo(), self.l2, prefactor=0.5)
        r_y1 -= ndot('mi,ma->ia', self.build_x1l1oo(self.x1,self.l1), self.Hov)	# q
        r_y1 -= ndot('ie,ea->ia', self.Hov, self.build_x1l1vv(self.x1,self.l1))	# q
        r_y1 -= ndot('mn,mina->ia', self.build_x1l1oo(self.x1,self.l1), self.Hooov, prefactor=2.0)	# q
        r_y1 -= ndot('mn,imna->ia', self.build_x1l1oo(self.x1,self.l1), self.Hooov, prefactor=-1.0)	# q
        r_y1 -= ndot('mena,imne->ia', self.build_x1l1ovov(self.x1,self.l1), self.Hooov, prefactor=2.0)	# q
        r_y1 -= ndot('mena,mine->ia', self.build_x1l1ovov(self.x1,self.l1), self.Hooov, prefactor=-1.0)	# q
        r_y1 += ndot('meif,fmae->ia', self.build_x1l1ovov(self.x1,self.l1), self.Hvovv, prefactor=2.0)	# q
        r_y1 += ndot('meif,fmea->ia', self.build_x1l1ovov(self.x1,self.l1), self.Hvovv, prefactor=-1.0)	# q
        r_y1 += ndot('ef,fiea->ia', self.build_x1l1vv(self.x1,self.l1), self.Hvovv, prefactor=2.0)	# q
        r_y1 += ndot('ef,fiae->ia', self.build_x1l1vv(self.x1,self.l1), self.Hvovv, prefactor=-1.0)	# q
        r_y1 += ndot('ianf,nf->ia', self.build_Lx2ovov(self.get_L('oovv'),self.x2), self.l1)	# r
        r_y1 -= ndot('ni,na->ia', self.build_Goo(self.x2, self.get_L('oovv')), self.l1)	# r
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Gvv(self.x2, self.get_L('oovv')))	# r Gvv is alreay negative
        r_y1 -= ndot('mnif,mfna->ia', self.build_x1l2ooov(self.x1,self.l2), self.Hovov)	# s
        r_y1 -= ndot('ifne,enaf->ia', self.Hovov, self.build_x1l2vovv(self.x1,self.l2))	# s
        r_y1 -= ndot('minf,mfan->ia', self.build_x1l2ooov(self.x1,self.l2), self.Hovvo)	# s
        r_y1 -= ndot('ifen,enfa->ia', self.Hovvo, self.build_x1l2vovv(self.x1,self.l2))	# s
        r_y1 += ndot('fgae,eifg->ia', self.Hvvvv, self.build_x1l2vovv(self.x1,self.l2), prefactor=0.5)	# s
        r_y1 += ndot('fgea,eigf->ia', self.Hvvvv, self.build_x1l2vovv(self.x1,self.l2), prefactor=0.5)	# s
        r_y1 += ndot('imno,mona->ia', self.Hoooo, self.build_x1l2ooov(self.x1,self.l2), prefactor=0.5)	# s
        r_y1 += ndot('mino,mnoa->ia', self.Hoooo, self.build_x1l2ooov(self.x1,self.l2), prefactor=0.5)	# s

	## 3-body terms

        tmp  =  ndot('nb,fb->nf', self.x1, self.build_Gvv(self.t2, self.l2))  
        r_y1 += ndot('inaf,nf->ia', self.get_L('oovv'), tmp)  # Gvv already negative
        tmp  =  ndot('me,fa->mefa', self.x1, self.build_Gvv(self.t2, self.l2))  
        r_y1 += ndot('mief,mefa->ia', self.get_L('oovv'), tmp)  
        tmp  =  ndot('me,ni->meni', self.x1, self.build_Goo(self.t2, self.l2))  
        r_y1 -= ndot('meni,mnea->ia', tmp, self.get_L('oovv'))  
        tmp  =  ndot('jf,nj->fn', self.x1, self.build_Goo(self.t2, self.l2))  
        r_y1 -= ndot('inaf,fn->ia', self.get_L('oovv'), tmp)  

	## 3-body terms over

	## X2 * L2 terms 

        r_y1  -=  ndot('mi,ma->ia', self.build_Goo(self.x2, self.l2), self.Hov)  # t
        r_y1  +=  ndot('ie,ea->ia', self.Hov, self.build_Gvv(self.x2, self.l2))  # t
        r_y1  -=  ndot('igne,gnea->ia', self.build_l2x2ovov_1(self.l2, self.x2), self.Hvovv)  # t
        r_y1  -=  ndot('igne,gnae->ia', self.build_l2x2ovov_2(self.l2, self.x2), self.Hvovv)  # t
        r_y1  -=  ndot('gief,gaef->ia', self.Hvovv, self.build_l2x2vvvv(self.l2, self.x2))  # t
        r_y1  +=  ndot('igme,gmae->ia', self.build_l2x2ovov_3(self.l2, self.x2), self.Hvovv, prefactor=2.0) # t
        r_y1  +=  ndot('igme,gmea->ia', self.build_l2x2ovov_3(self.l2, self.x2), self.Hvovv, prefactor=-1.0) # t
        r_y1  -=  ndot('giea,ge->ia', self.Hvovv, self.build_Gvv(self.l2, self.x2), prefactor=2.0)  # t
        r_y1  -=  ndot('giae,ge->ia', self.Hvovv, self.build_Gvv(self.l2, self.x2), prefactor=-1.0)  # t
        r_y1  +=  ndot('oimn,mnoa->ia', self.build_l2x2oooo(self.l2, self.x2), self.Hooov)  # t
        r_y1  +=  ndot('inoe,oane->ia', self.Hooov, self.build_l2x2ovov_2(self.l2, self.x2))  # t
        r_y1  +=  ndot('miof,oamf->ia', self.Hooov, self.build_l2x2ovov_1(self.l2, self.x2))  # t
        r_y1  -=  ndot('mioa,mo->ia', self.Hooov, self.build_Goo(self.x2, self.l2), prefactor=2.0)  # t
        r_y1  -=  ndot('imoa,mo->ia', self.Hooov, self.build_Goo(self.x2, self.l2), prefactor=-1.0)  # t
        r_y1  -=  ndot('imoe,oame->ia', self.Hooov, self.build_l2x2ovov_3(self.l2, self.x2), prefactor=2.0) # t
        r_y1  -=  ndot('mioe,oame->ia', self.Hooov, self.build_l2x2ovov_3(self.l2, self.x2), prefactor=-1.0) # t


	# y1 over !! 

	# Homogenous terms of Y2 equations 
        
        r_y2 = 0.5 * self.omega * self.y2.copy()
        r_y2 += ndot('ia,jb->ijab', self.y1, self.Hov, prefactor=2.0)
        r_y2 -= ndot('ja,ib->ijab', self.y1, self.Hov)
        r_y2 += ndot('ijeb,ea->ijab', self.y2, self.Hvv)
        r_y2 -= ndot('im,mjab->ijab', self.Hoo, self.y2)
        r_y2 += ndot('ijmn,mnab->ijab', self.Hoooo, self.y2, prefactor=0.5)
        r_y2 += ndot('ijef,efab->ijab', self.y2, self.Hvvvv, prefactor=0.5)
        r_y2 += ndot('ie,ejab->ijab', self.y1, self.Hvovv, prefactor=2.0)
        r_y2 += ndot('ie,ejba->ijab', self.y1, self.Hvovv, prefactor=-1.0)
        r_y2 -= ndot('mb,jima->ijab', self.y1, self.Hooov, prefactor=2.0)
        r_y2 -= ndot('mb,ijma->ijab', self.y1, self.Hooov, prefactor=-1.0)
        r_y2 += ndot('ieam,mjeb->ijab', self.Hovvo, self.y2, prefactor=2.0)
        r_y2 += ndot('iema,mjeb->ijab', self.Hovov, self.y2, prefactor=-1.0)
        r_y2 -= ndot('mibe,jema->ijab', self.y2, self.Hovov)
        r_y2 -= ndot('mieb,jeam->ijab', self.y2, self.Hovvo)
        r_y2 += ndot('ijeb,ae->ijab', self.get_L('oovv'), self.build_Gvv(self.y2, self.t2))
        r_y2 -= ndot('mi,mjab->ijab', self.build_Goo(self.t2, self.y2), self.get_L('oovv'))

	# InHomogenous terms of Y2 equations 

        r_y2 += ndot('ia,jb->ijab', self.l1, self.build_Aov(), prefactor=2.0) # o
        r_y2 -= ndot('ja,ib->ijab', self.l1, self.build_Aov()) # o
        r_y2 += ndot('ijeb,ea->ijab', self.l2, self.build_Avv()) # p
        r_y2 -= ndot('im,mjab->ijab', self.build_Aoo(), self.l2) # p

        r_y2 -= ndot('mieb,meja->ijab', self.get_L('oovv'), self.build_x1l1ovov(self.x1, self.l1)) # u
        r_y2 -= ndot('ijae,eb->ijab', self.get_L('oovv'), self.build_x1l1vv(self.x1, self.l1)) # u
        r_y2 -= ndot('mi,jmba->ijab', self.build_x1l1oo(self.x1, self.l1), self.get_L('oovv')) # u
        r_y2 += ndot('imae,mejb->ijab', self.get_L('oovv'), self.build_x1l1ovov(self.x1, self.l1), prefactor=2.0) # u

        r_y2 -= ndot('mijb,ma->ijab', self.build_x1l2ooov(self.x1, self.l2), self.Hov) # w
        r_y2 -= ndot('ie,ejba->ijab', self.Hov, self.build_x1l2vovv(self.x1, self.l2)) # w

        r_y2 -= ndot('mijf,fmba->ijab', self.build_x1l2ooov(self.x1, self.l2), self.Hvovv) # w
        r_y2 -= ndot('fjea,eibf->ijab', self.Hvovv, self.build_x1l2vovv(self.x1, self.l2)) # w
        r_y2 -= ndot('fibe,ejfa->ijab', self.Hvovv, self.build_x1l2vovv(self.x1, self.l2)) # w
        r_y2 += ndot('ijfb,fa->ijab', self.l2, self.Hvovvx1vv(self.Hvovv, self.x1)) # w
        r_y2 += ndot('fiea,ejbf->ijab', self.Hvovv, self.build_x1l2vovv(self.x1, self.l2), prefactor=2.0) # w
        r_y2 += ndot('fiae,ejbf->ijab', self.Hvovv, self.build_x1l2vovv(self.x1, self.l2), prefactor=-1.0) # w

        r_y2 += ndot('minb,jmna->ijab', self.build_x1l2ooov(self.x1, self.l2), self.Hooov) # w
        r_y2 += ndot('mnib,mjna->ijab', self.build_x1l2ooov(self.x1, self.l2), self.Hooov) # w
        r_y2 += ndot('jine,enba->ijab', self.Hooov, self.build_x1l2vovv(self.x1, self.l2)) # w
        r_y2 -= ndot('mina,mnjb->ijab', self.Hooov, self.build_x1l2ooov(self.x1, self.l2), prefactor=2.0) # w
        r_y2 -= ndot('imna,mnjb->ijab', self.Hooov, self.build_x1l2ooov(self.x1, self.l2), prefactor=-1.0) # w
        r_y2 -= ndot('in,jnba->ijab', self.Hooovx1oo(self.Hooov, self.x1), self.l2) # w

        #
        r_y2 += ndot('ijmn,mnab->ijab', self.build_l2x2oooo(self.l2, self.x2), self.get_MO('oovv'), prefactor=0.5) # x
        r_y2 += ndot('ibne,jnae->ijab', self.build_l2x2ovov_2(self.l2, self.x2), self.get_MO('oovv'), prefactor=0.5) # x same.2
        r_y2 += ndot('ibne,njae->ijab', self.build_l2x2ovov_1(self.l2, self.x2), self.get_MO('oovv'), prefactor=0.5) # x same.1
        r_y2 += ndot('ibne,jnea->ijab', self.build_l2x2ovov_1(self.l2, self.x2), self.get_MO('oovv'), prefactor=0.5) # x same.1
        r_y2 += ndot('ibne,njea->ijab', self.build_l2x2ovov_2(self.l2, self.x2), self.get_MO('oovv'), prefactor=0.5) # x same.2
        r_y2 += ndot('ijfe,baef->ijab', self.get_MO('oovv'), self.build_l2x2vvvv(self.l2, self.x2), prefactor=0.5) # x

        r_y2 -= ndot('inae,jbne->ijab', self.get_L('oovv'), self.build_l2x2ovov_2(self.l2, self.x2), prefactor=1.0) # x 
        r_y2 -= ndot('in,jnba->ijab', self.build_Goo(self.get_L('oovv'), self.x2), self.l2, prefactor=1.0) # x 
        r_y2 += ndot('ijfb,af->ijab', self.l2, self.build_Gvv(self.get_L('oovv'), self.x2), prefactor=1.0) # x 

        r_y2 += ndot('ijae,be->ijab', self.get_L('oovv'), self.build_Gvv(self.l2, self.x2), prefactor=1.0) # x 
        r_y2 -= ndot('imab,jm->ijab', self.get_L('oovv'), self.build_Goo(self.l2, self.x2), prefactor=1.0) # x 
        r_y2 -= ndot('ibme,mjea->ijab', self.build_l2x2ovov_3(self.l2, self.x2), self.get_L('oovv'), prefactor=1.0) # x 


        r_y2 += ndot('imae,jbme->ijab', self.get_L('oovv'), self.build_l2x2ovov_3(self.l2, self.x2), prefactor=2.0) # x 


        self.y1 += r_y1/self.Dia
        tmp = r_y2
        tmp += r_y2.swapaxes(0,1).swapaxes(2,3)
        self.y2 += tmp/self.Dijab







        #tmp   = ndot('ma,me->ae', self.Hov, self.x1)
        #r_y1 -= ndot('ie,ae->ia', self.l1, tmp)

        #tmp   = ndot('ie,me->im', self.Hov, self.x1)
        #r_y1 -= ndot('im,ma->ia', tmp, self.l1)

        #tmp   = ndot('me,ne->mn', self.x1, self.l1)
        #r_y1 -= ndot('mina,mn->ia', self.Hooov, tmp, prefactor=2.0)
        #r_y1 -= ndot('imna,mn->ia', self.Hooov, tmp, prefactor=-1.0)

        #tmp   = ndot('me,na->mnea', self.x1, self.l1)
        #r_y1 -= ndot('imne,mnea->ia', self.Hooov, tmp, prefactor=2.0)
        #r_y1 -= ndot('mine,mnea->ia', self.Hooov, tmp, prefactor=-1.0)

        #tmp   = ndot('me,ne->mn', self.x1, self.l1)
        #r_y1 -= ndot('mina,mn->ia', self.Hooov, tmp, prefactor=2.0)
        #r_y1 -= ndot('imna,mn->ia', self.Hooov, tmp, prefactor=-1.0)

        #tmp   = ndot('me,na->mnea', self.x1, self.l1)
        #r_y1 -= ndot('imne,mnea->ia', self.Hooov, tmp, prefactor=2.0)
        #r_y1 -= ndot('mine,mnea->ia', self.Hooov, tmp, prefactor=-1.0)



    def pseudoresponse(self, hand):
        polar1 = 0
        polar2 = 0
        if hand == 'right':
            z1 = self.x1 ; z2 = self.x2
        else:
            z1 = self.y1 ; z2 = self.y2
        polar1 += ndot('ia,ai->', z1, self.build_Avo(), prefactor=2.0)
        polar2 += ndot('ijab,abij->', z2, self.build_Avvoo(), prefactor=4.0)
        polar2 += ndot('ijba,abij->', z2, self.build_Avvoo(), prefactor=-2.0)
        return -2.0 * (polar1 + polar2) 

 

    def solve(self, hand, r_conv=1.e-13, maxiter=50, max_diis=8):
        ### Setup DIIS
        if hand == 'right':
            z1 = self.x1 ; z2 = self.x2
        else:
            z1 = self.y1 ; z2 = self.y2

        diis_vals_z1 = [z1.copy()]
        diis_vals_z2 = [z2.copy()]
        diis_errors = []

        ### Start Iterations
        ccresponse_tstart = time.time()
        pseudoresponse_old = self.pseudoresponse(hand)
        print("CCRESPONSE Iteration %3d: pseudoresponse = %.15f   dE = % .5E " % (0, pseudoresponse_old, -pseudoresponse_old))
        #print('\nAvo\n')
        #print(self.build_Avo())
        #print('\nAvvoo\n')
        #print(self.build_Avvoo())

        # Iterate!
        diis_size = 0
        for CCRESPONSE_iter in range(1, maxiter + 1):

            # Save new amplitudes
            oldz1 = z1.copy()
            oldz2 = z2.copy()
            if hand == 'right':
                self.update_X()
            else:
                self.update_Y()
            pseudoresponse = self.pseudoresponse(hand)

            # Print CCRESPONSE iteration information
            print('CCRESPONSE Iteration %3d: pseudoresponse = %.15f   dE = % .5E   DIIS = %d' % (CCRESPONSE_iter, pseudoresponse, (pseudoresponse - pseudoresponse_old), diis_size))

            # Check convergence
            if (abs(pseudoresponse - pseudoresponse_old) < r_conv):
                print('\nCCRESPONSE has converged in %.3f seconds!' % (time.time() - ccresponse_tstart))
                return pseudoresponse

            # Add DIIS vectors
            diis_vals_z1.append(z1.copy())
            diis_vals_z2.append(z2.copy())

            # Build new error vector
            error_z1 = (diis_vals_z1[-1] - oldz1).ravel()
            error_z2 = (diis_vals_z2[-1] - oldz2).ravel()
            diis_errors.append(np.concatenate((error_z1, error_z2)))

            # Update old energy
            pseudoresponse_old = pseudoresponse

            if CCRESPONSE_iter >= 1:
                # Limit size of DIIS vector
                if (len(diis_vals_z1) > max_diis):
                    del diis_vals_z1[0]
                    del diis_vals_z2[0]
                    del diis_errors[0]

                diis_size = len(diis_vals_z1) - 1

                # Build error matrix B
                B = np.ones((diis_size + 1, diis_size + 1)) * -1
                B[-1, -1] = 0

                for n1, e1 in enumerate(diis_errors):
                    B[n1, n1] = np.dot(e1, e1)
                    for n2, e2 in enumerate(diis_errors):
                        if n1 >= n2: continue
                        B[n1, n2] = np.dot(e1, e2)
                        B[n2, n1] = B[n1, n2]

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

                # Build residual vector
                resid = np.zeros(diis_size + 1)
                resid[-1] = -1

                # Solve pulay equations
                ci = np.linalg.solve(B, resid)

                # Calculate new amplitudes
                z1[:] = 0
                z2[:] = 0
                for num in range(diis_size):
                    z1 += ci[num] * diis_vals_z1[num + 1]
                    z2 += ci[num] * diis_vals_z2[num + 1]




# End CCRESPONSE class

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
