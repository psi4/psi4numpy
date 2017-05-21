'''
Created on Jan 24, 2015

@author: tianyuanzhang
'''
# from sp_integrals.AntisymmetrizedTwoElectronIntegrals import AntiSymTwoElectronInt
# from sp_math.PsithonMath import spat2SpinMatrix
from itertools import combinations
from Determinant import Determinant_bits
from scipy.sparse import csr_matrix
import numpy
import time
import sys

class MatrixElements:
    '''
    class for Full CI matrix elements
    '''
    def __init__(self, nmo, ndocc, H_spin, mo_spin_eri):
        """Constructor for MatrixElements"""
#         self.rhf = rhf
#         self.antiSym2eInt = AntiSymTwoElectronInt.createFrom2eIntSpatial(self.rhf.calc2eIntMO())
#         self.Hspin = spat2SpinMatrix(self.rhf.calcCoreHamiltonianMO())
        self.numberOfBasis = nmo
        self.numOCC = ndocc
        self.Hspin = H_spin
        self.antiSym2eInt = mo_spin_eri
        
        self.detList = None
    
    def generateDeterminants(self):
        """Construct a list of all the possible determinants"""
        self.detList = []
        for alpha in combinations(xrange(self.numberOfBasis), self.numOCC):
            for beta in combinations(xrange(self.numberOfBasis), self.numOCC):
                self.detList.append(Determinant_bits(alphaObtList=alpha, betaObtList=beta))
        return self.detList
    
    def generateMatrix(self):
        """Generate CI Matrix"""
        self.generateDeterminants()
        numDet = len(self.detList)
        iList, jList, vList = [],[],[]
        for i in xrange(numDet):
            for j in xrange(numDet):
                value = self.calcMatrixElement(self.detList[i], self.detList[j])
                if abs(value)>1e-14:
                    iList.append(i)
                    jList.append(j)
                    vList.append(value)
        return csr_matrix((vList,(iList,jList)),shape=(numDet,numDet))
    
    def generateMatrixWithDets(self, detList):
        """Generate CI Matrix"""
        self.detList = detList
        numDet = len(self.detList)
        iList, jList, vList = [],[],[]
        for i in xrange(numDet):
            for j in xrange(numDet):
                value = self.calcMatrixElement(self.detList[i], self.detList[j])
                if abs(value)>1e-14:
                    iList.append(i)
                    jList.append(j)
                    vList.append(value)
        return csr_matrix((vList,(iList,jList)),shape=(numDet,numDet))
    
    def generateMatrix_test(self):
        self.generateDeterminants()
        numDet = len(self.detList)
        print "The full CI calculation involves",numDet,"determinants."
        iList, jList, vList = [],[],[]
        print "Finished row calculations:\n"
        t0 = time.time()
        tPre = t0
        for i in xrange(numDet):
            for j in xrange(numDet):
                value = self.calcMatrixElement(self.detList[i], self.detList[j])
                if abs(value)>1e-14:
                    iList.append(i)
                    jList.append(j)
                    vList.append(value)
            t = time.time()
            if t-tPre>1:
                tRemain = (numDet-i-1)/(1.+i)*(t-t0)
                sys.stdout.write("\r"+str(i)+"/"+str(numDet)+" time remain:"+str(tRemain)+"s")
                tPre = t
        print "\nThere are totally",len(vList),"matrix elements that is not zero."
        return csr_matrix((vList,(iList,jList)),shape=(numDet,numDet))
    
    def calcMatrixElement(self, det1, det2):
        """Calculate a matrix element by two determinants"""
        numUniqueOrbitals = None
        if det1.diff2OrLessOrbitals(det2):
            numUniqueOrbitals = det1.numberOfTotalDiffOrbitals(det2)
            if numUniqueOrbitals == 2:
                return self.calcMatrixElementDiffIn2(det1, det2)
            elif numUniqueOrbitals == 1:
                return self.calcMatrixElementDiffIn1(det1, det2)
            else:
                return self.calcMatrixElementIdentialDet(det1)
        else:
            return 0.0
    
    def calcMatrixElementDiffIn2(self, det1, det2):
        """Calculate a matrix element by two determinants where the determinants differ by 2 spin orbitals"""
        unique1, unique2, sign = det1.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
        return sign * self.antiSym2eInt[unique1[0], unique1[1], unique2[0], unique2[1]]
    
    def calcMatrixElementDiffIn1(self, det1, det2):
        """Calculate a matrix element by two determinants where the determinants differ by 1 spin orbitals"""
        unique1, unique2, sign = det1.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
        m = unique1[0]
        p = unique2[0]
        Helem = self.Hspin[m,p]
        common = det1.getCommonOrbitalsInMixedSpinIndexList(det2)
        Relem = 0.0
        for n in common:
            Relem += self.antiSym2eInt[m,n,p,n]
        return sign * (Helem + Relem)
    
    def calcMatrixElementIdentialDet(self, det):
        """Calculate a matrix element by two determinants where they are identical"""
        spinObtList = det.getOrbitalMixedIndexList()
        Helem = 0.0
        for m in spinObtList:
            Helem += self.Hspin[m,m]
        length = len(spinObtList)
        Relem = 0.0
        for m in xrange(length-1):
            for n in xrange(m+1, length):
                Relem += self.antiSym2eInt[spinObtList[m], spinObtList[n], spinObtList[m], spinObtList[n]]
#                 Relem += self.antiSym2eInt.get2eInt(spinObtList[m], spinObtList[n], spinObtList[m], spinObtList[n])
        return Helem + Relem

class MatrixElements_dense(MatrixElements):
    '''
    class for Full CI matrix elements
    '''
    def __init__(self, nmo, ndocc, H_spin, mo_spin_eri):
        """Constructor for MatrixElements"""
        MatrixElements.__init__(self, nmo, ndocc, H_spin, mo_spin_eri)
    
    def generateMatrix(self):
        """Generate CI Matrix"""
        self.generateDeterminants()
        numDet = len(self.detList)
        matrix = numpy.zeros((numDet,numDet))
        for i in xrange(numDet):
            for j in xrange(i+1):
                matrix[i,j] = self.calcMatrixElement(self.detList[i], self.detList[j])
        return matrix
    
    def calcMatrixElement(self, det1, det2):
        """Calculate a matrix element by two determinants"""
        numUniqueOrbitals = None
        if det1.diff2OrLessOrbitals(det2):
            numUniqueOrbitals = det1.numberOfTotalDiffOrbitals(det2)
            if numUniqueOrbitals == 2:
                return self.calcMatrixElementDiffIn2(det1, det2)
            elif numUniqueOrbitals == 1:
                return self.calcMatrixElementDiffIn1(det1, det2)
            else:
                return self.calcMatrixElementIdentialDet(det1)
        else:
            return 0.0

if __name__ == '__main__':
    pass