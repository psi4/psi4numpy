class helper_cclambda(object):

    def __init__(self, ccsd, hbar):

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nocc = ccsd.ndocc
        self.nvirt = ccsd.nmo - ccsd.nocc

        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}

        self.F = ccsd.F
        self.Dia = ccsd.Dia
        self.Dijab = ccsd.Dijab
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2

        self.ttau  =  hbar.ttau
        self.Loovv =  hbar.Loovv
        self.Looov =  hbar.Looov
        self.Lvovv =  hbar.Lvovv
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

        self.l1 = 2.0 * self.t1
        tmp = self.t2
        self.l2 = 2.0 * (2.0 * tmp - tmp.swapaxes(2,3))

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


    def build_Goo(self):
        self.Goo = 0
        self.Goo += ndot('mjab,ijab->mi', self.t2, self.l2)
        return self.Goo

    def build_Gvv(self):
        self.Gvv = 0
        self.Gvv -= ndot('ijab,ijeb->ae', self.l2, self.t2)
        return self.Gvv

    def update(self):
        r_l1  = 2.0 * self.Hov.copy()
        r_l1 += ndot('ie,ea->ia', self.l1, self.Hvv)
        r_l1 -= ndot('im,ma->ia', self.Hoo, self.l1)
        r_l1 += ndot('ieam,me->ia', self.Hovvo, self.l1, prefactor=2.0)
        r_l1 += ndot('iema,me->ia', self.Hovov, self.l1, prefactor=-1.0)
        r_l1 += ndot('imef,efam->ia', self.l2, self.Hvvvo)
        r_l1 -= ndot('iemn,mnae->ia', self.Hovoo, self.l2)
        r_l1 -= ndot('eifa,ef->ia', self.Hvovv, self.build_Gvv(), prefactor=2.0)
        r_l1 -= ndot('eiaf,ef->ia', self.Hvovv, self.build_Gvv(), prefactor=-1.0)
        r_l1 -= ndot('mina,mn->ia', self.Hooov, self.build_Goo(), prefactor=2.0)
        r_l1 -= ndot('imna,mn->ia', self.Hooov, self.build_Goo(), prefactor=-1.0)


        r_l2 = self.Loovv.copy()
        r_l2 += ndot('ia,jb->ijab', self.l1, self.Hov, prefactor=2.0)
        r_l2 -= ndot('ja,ib->ijab', self.l1, self.Hov)
        r_l2 += ndot('ijeb,ea->ijab', self.l2, self.Hvv)
        r_l2 -= ndot('im,mjab->ijab', self.Hoo, self.l2)
        r_l2 += ndot('ijmn,mnab->ijab', self.Hoooo, self.l2, prefactor=0.5)
        r_l2 += ndot('ijef,efab->ijab', self.l2, self.Hvvvv, prefactor=0.5)
        r_l2 += ndot('ie,ejab->ijab', self.l1, self.Hvovv, prefactor=2.0)
        r_l2 += ndot('ie,ejba->ijab', self.l1, self.Hvovv, prefactor=-1.0)
        r_l2 -= ndot('mb,jima->ijab', self.l1, self.Hooov, prefactor=2.0)
        r_l2 -= ndot('mb,ijma->ijab', self.l1, self.Hooov, prefactor=-1.0)
        r_l2 += ndot('ieam,mjeb->ijab', self.Hovvo, self.l2, prefactor=2.0)
        r_l2 += ndot('iema,mjeb->ijab', self.Hovov, self.l2, prefactor=-1.0)
        r_l2 -= ndot('mibe,jema->ijab', self.l2, self.Hovov)
        r_l2 -= ndot('mieb,jeam->ijab', self.l2, self.Hovvo)
        r_l2 += ndot('ijeb,ae->ijab', self.Loovv, self.build_Gvv())
        r_l2 -= ndot('mi,mjab->ijab', self.build_Goo(), self.Loovv)

        self.l1 += r_l1/self.Dia

        old_l2 = self.l2

        self.l2 += r_l2/self.Dijab 
        self.l2 += (r_l2/self.Dijab).swapaxes(0,1).swapaxes(2,3) 

        rms = 2.0 * np.einsum('ia,ia->', r_l1/self.Dia, r_l1/self.Dia) 
        rms += np.einsum('ijab,ijab->', old_l2 - self.l2, old_l2 - self.l2) 
        return np.sqrt(rms)
   



    def pseudoenergy(self):
        pseudoenergy = 0
        pseudoenergy += ndot('ijab,ijab->', self.get_MO('oovv'), self.l2, prefactor=0.5)
        return pseudoenergy



    def compute_lambda(self, r_conv=1e-7, maxiter=100, max_diis=8):
        print('\n Solving lambda equations ...\n')
        ### Setup DIIS
        diis_vals_l1 = [self.l1.copy()]
        diis_vals_l2 = [self.l2.copy()]
        diis_errors = []

        ### Start Iterations
        cclambda_tstart = time.time()

        pseudoenergy_old = self.pseudoenergy()
        print("CCLAMBDA Iteration %3d: pseudoenergy = %.15f   dE = % .5E   MP2" % (0, pseudoenergy_old, -pseudoenergy_old))

        # Iterate!
        diis_size = 0
        for CCLAMBDA_iter in range(1, maxiter + 1):

            # Save new amplitudes
            oldl1 = self.l1.copy()
            oldl2 = self.l2.copy()

            rms = self.update()

            # Compute lambda 
            pseudoenergy = self.pseudoenergy()

            # Print CCLAMBDA iteration information
            print('CCLAMBDA Iteration %3d: pseudoenergy = %.15f   dE = % .5E   DIIS = %d' % (CCLAMBDA_iter, pseudoenergy, (pseudoenergy - pseudoenergy_old), diis_size))

            # Check convergence
            #if (abs(pseudoenergy - pseudoenergy_old) < r_conv):
            if (rms < r_conv):
                print('\nCCLAMBDA has converged in %.3f seconds!' % (time.time() - cclambda_tstart))
                #print(self.l1)
                #print(self.l2)
                return pseudoenergy

            # Add DIIS vectors
            diis_vals_l1.append(self.l1.copy())
            diis_vals_l2.append(self.l2.copy())

            # Build new error vector
            error_l1 = (diis_vals_l1[-1] - oldl1).ravel()
            error_l2 = (diis_vals_l2[-1] - oldl2).ravel()
            diis_errors.append(np.concatenate((error_l1, error_l2)))

            # Update old energy
            pseudoenergy_old = pseudoenergy

            if CCLAMBDA_iter >= 1:
               # Limit size of DIIS vector
                if (len(diis_vals_l1) > max_diis):
                    del diis_vals_l1[0]
                    del diis_vals_l2[0]
                    del diis_errors[0]

                diis_size = len(diis_vals_l1) - 1

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
                self.l1[:] = 0
                self.l2[:] = 0
                for num in range(diis_size):
                    self.l1 += ci[num] * diis_vals_l1[num + 1]
                    self.l2 += ci[num] * diis_vals_l2[num + 1]

