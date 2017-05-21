'''
Created on Jan 10, 2015

@author: tianyuanzhang
'''
from itertools import combinations
class Determinant:
    '''
    class for determinant
    '''

class Determinant_bits(Determinant):
    '''
    class for determinant stored in bits
    '''

    def __init__(self, nmo=0, alphaObtBits=0, betaObtBits=0, alphaObtList=None, betaObtList=None, str=None):
        """Constructor for Determinant_bits"""
        if alphaObtBits == 0 and alphaObtList != None:
            alphaObtBits = Determinant_bits.obtIndexList2ObtBits(alphaObtList)
        if betaObtBits == 0 and betaObtList != None:
            betaObtBits = Determinant_bits.obtIndexList2ObtBits(betaObtList)
        self.nmo = nmo
        self.alphaObtBits = alphaObtBits
        self.betaObtBits = betaObtBits
    
    def getNumOrbitals(self):
        """Return the number of orbitals (alpha, beta) in this determinant"""
        return Determinant_bits.countNumOrbitalsInBits(self.alphaObtBits), Determinant_bits.countNumOrbitalsInBits(self.betaObtBits)
    
    def getOrbitalIndexLists(self):
        """Return lists of orbital index"""
        return Determinant_bits.obtBits2ObtIndexList(self.alphaObtBits), Determinant_bits.obtBits2ObtIndexList(self.betaObtBits)
    
    def getOrbitalMixedIndexList(self):
        """Return lists of orbital in mixed spin index"""
        return Determinant_bits.obtBits2ObtMixSpinIndexList(self.alphaObtBits, self.betaObtBits)
    
    @staticmethod
    def countNumOrbitalsInBits(bits):
        """Return the number of orbitals in this bits"""
        count = 0
        while bits!=0:
            if bits&1==1:
                count += 1
            bits >>= 1
        return count
    
    @staticmethod
    def countNumOrbitalsInBitsUpTo6(bits):
        """Return the number of orbitals in this bits"""
        count = 0
        while bits!=0 and count<6:
            if bits&1==1:
                count += 1
            bits >>= 1
        return count
    
    @staticmethod
    def obtBits2ObtIndexList(bits):
        """Return the corresponding list of orbital numbers of orbital bits"""
        i = 0
        obts = []
        while bits!=0:
            if bits&1==1:
                obts.append(i)
            bits >>= 1
            i += 1
        return obts
    
    @staticmethod
    def mixIndexList(alphaList, betaList):
        """Mix the alpha and beta orbital index list to one mixed list"""
        return [elem*2 for elem in alphaList]+[elem*2+1 for elem in betaList]
    
    @staticmethod
    def obtBits2ObtMixSpinIndexList(alphaBits, betaBits):
        """Return the corresponding list of orbital numbers of orbital bits"""
        alphaList, betaList = Determinant_bits.obtBits2ObtIndexList(alphaBits), Determinant_bits.obtBits2ObtIndexList(betaBits)
        return Determinant_bits.mixIndexList(alphaList, betaList)
    
    @staticmethod
    def obtIndexList2ObtBits(obtList):
        """Return the corresponding orbital bits of list of orbital numbers"""
        if len(obtList)==0:
            return 0
        obtList = sorted(obtList, reverse = True)
        iPre = obtList[0]
        bits = 1
        for i in obtList:
            bits <<= iPre-i
            bits |= 1
            iPre = i
        bits <<= iPre
        return bits
    
    @staticmethod
    def getOrbitalPositions(bits, orbitalIndexList):
        """Return the position of orbital in determinant"""
        count = 0
        index = 0
        positions = []
        for i in orbitalIndexList:
            while index<i:
                if bits&1==1:
                    count += 1
                bits >>= 1
                index += 1
            positions.append(count)
            continue
        return positions
    
    def getOrbitalPositionLists(self, alphaIndexList, betaIndexList):
        """Return the positions of indexes in lists"""
        return Determinant_bits.getOrbitalPositions(self.alphaObtBits, alphaIndexList), Determinant_bits.getOrbitalPositions(self.betaObtBits, betaIndexList)
    
    def addAlphaOrbital(self, orbitalIndex):
        """Add an alpha orbital to the determinant"""
        self.alphaObtBits |= 1<<orbitalIndex
    
    def addBetaOrbital(self, orbitalIndex):
        """Add an beta orbital to the determinant"""
        self.betaObtBits |= 1<<orbitalIndex
    
    def removeAlphaOrbital(self, orbitalIndex):
        """remove an alpha orbital from the determinant"""
        self.alphaObtBits &= ~(1<<orbitalIndex)
    
    def removeBetaOrbital(self, orbitalIndex):
        """remove an beta orbital from the determinant"""
        self.betaObtBits &= ~(1<<orbitalIndex)
        
    def numberOfCommonOrbitals(self, another):
        """Return the number of common orbitals between this determinant and another determinant"""
        return Determinant_bits.countNumOrbitalsInBits(self.alphaObtBits & another.alphaObtBits), Determinant_bits.countNumOrbitalsInBits(self.betaObtBits & another.betaObtBits)
    
    def getCommonOrbitalsInLists(self, another):
        """Return common orbitals between this determinant and another determinant in lists"""
        return Determinant_bits.obtBits2ObtIndexList(self.alphaObtBits & another.alphaObtBits), Determinant_bits.obtBits2ObtIndexList(self.betaObtBits & another.betaObtBits)
    
    def getCommonOrbitalsInMixedSpinIndexList(self, another):
        alphaList, betaList = self.getCommonOrbitalsInLists(another)
        return Determinant_bits.mixIndexList(alphaList, betaList)
    
    def numberOfDiffOrbitals(self, another):
        """Return the number of different alpha and beta orbitals between this determinant and another determinant"""
        diffAlpha, diffBeta = Determinant_bits.countNumOrbitalsInBits(self.alphaObtBits ^ another.alphaObtBits), Determinant_bits.countNumOrbitalsInBits(self.betaObtBits ^ another.betaObtBits)
        return diffAlpha/2, diffBeta/2
    
    def numberOfTotalDiffOrbitals(self, another):
        """Return the number of different orbitals between this determinant and another determinant"""
        diffAlpha, diffBeta = self.numberOfDiffOrbitals(another)
        return diffAlpha+diffBeta
    
    def diff2OrLessOrbitals(self, another):
        """Return true if two determinants differ 2 or less orbitals"""
        diffAlpha, diffBeta = Determinant_bits.countNumOrbitalsInBitsUpTo6(self.alphaObtBits ^ another.alphaObtBits), Determinant_bits.countNumOrbitalsInBitsUpTo6(self.betaObtBits ^ another.betaObtBits)
        return (diffAlpha+diffBeta) <= 4
    
    @staticmethod
    def uniqueOrbitalsInBits(bits1, bits2):
        """Return the unique bits in two different bits"""
        common = bits1 & bits2
        return bits1^common, bits2^common
    
    @staticmethod
    def uniqueOrbitalsInLists(bits1, bits2):
        """Return the unique bits in two different bits"""
        bits1, bits2 = Determinant_bits.uniqueOrbitalsInBits(bits1, bits2)
        return Determinant_bits.obtBits2ObtIndexList(bits1), Determinant_bits.obtBits2ObtIndexList(bits2)
    
    def getUniqueOrbitalsInLists(self, another):
        """Return the unique orbital lists in two different determinants"""
        alphaList1, alphaList2 = Determinant_bits.uniqueOrbitalsInLists(self.alphaObtBits, another.alphaObtBits)
        betaList1, betaList2 = Determinant_bits.uniqueOrbitalsInLists(self.betaObtBits, another.betaObtBits)
        return (alphaList1, betaList1),(alphaList2, betaList2)
    
    def getUnoccupiedOrbitalsInLists(self, numOfBasis):
        """Return the unoccupied orbitals in the determinants"""
        alphaBits = ~self.alphaObtBits
        betaBits = ~self.betaObtBits
        alphaObts = []
        betaObts = []
        for i in xrange(numOfBasis):
            if alphaBits&1==1:
                alphaObts.append(i)
            alphaBits >>= 1
            if betaBits&1==1:
                betaObts.append(i)
            betaBits >>= 1
        return alphaObts, betaObts
    
    def getSignToMoveOrbitalsToTheFront(self, alphaIndexList, betaIndexList):
        """Return the final sign if move listed orbitals to the front"""
        sign = 1
        alphaPositions, betaPositions = self.getOrbitalPositionLists(alphaIndexList, betaIndexList)
        for i in xrange(len(alphaPositions)):
            if (alphaPositions[i]-i)%2==1:
                sign = -sign
        for i in xrange(len(betaPositions)):
            if (betaPositions[i]-i)%2==1:
                sign = -sign
        return sign
    
    def getUniqueOrbitalsInListsPlusSign(self, another):
        """Return the unique orbital lists in two different determinants and the sign of the maximum coincidence determinants"""
        alphaList1, alphaList2 = Determinant_bits.uniqueOrbitalsInLists(self.alphaObtBits, another.alphaObtBits)
        betaList1, betaList2 = Determinant_bits.uniqueOrbitalsInLists(self.betaObtBits, another.betaObtBits)
        sign1, sign2 = self.getSignToMoveOrbitalsToTheFront(alphaList1, betaList1), another.getSignToMoveOrbitalsToTheFront(alphaList2, betaList2)
        return (alphaList1, betaList1),(alphaList2, betaList2),sign1*sign2
    
    def getUniqueOrbitalsInMixIndexListsPlusSign(self, another):
        """Return the unique orbital lists in two different determinants and the sign of the maximum coincidence determinants"""
        (alphaList1, betaList1),(alphaList2, betaList2),sign = self.getUniqueOrbitalsInListsPlusSign(another)
        return Determinant_bits.mixIndexList(alphaList1, betaList1), Determinant_bits.mixIndexList(alphaList2, betaList2), sign
    
    def toIntTuple(self):
        """Return a int tuple"""
        return (self.alphaObtBits,self.betaObtBits)
    
    @staticmethod
    def createFromIntTuple(intTuple):
        return Determinant_bits(alphaObtBits=intTuple[0], betaObtBits=intTuple[1])
    
    def generateSingleAndDoubleExcitationOfDet(self, numOfBasis):
        """generate all the single and double excitations of determinant in a list"""
        alphaO, betaO = self.getOrbitalIndexLists()
        alphaU, betaU = self.getUnoccupiedOrbitalsInLists(numOfBasis)
        dets = []
        
        for i in alphaO:
            for j in alphaU:
                det = self.copy()
                det.removeAlphaOrbital(i)
                det.addAlphaOrbital(j)
                dets.append(det)

        for k in betaO:
            for l in betaU:
                det = self.copy()
                det.removeBetaOrbital(k)
                det.addBetaOrbital(l)
                dets.append(det)
                
        for i in alphaO:
            for j in alphaU:
                for k in betaO:
                    for l in betaU:
                        det = self.copy()
                        det.removeAlphaOrbital(i)
                        det.addAlphaOrbital(j)
                        det.removeBetaOrbital(k)
                        det.addBetaOrbital(l)
                        dets.append(det)
        
        for i1,i2 in combinations(alphaO,2):
            for j1,j2 in combinations(alphaU,2):
                det = self.copy()
                det.removeAlphaOrbital(i1)
                det.addAlphaOrbital(j1)
                det.removeAlphaOrbital(i2)
                det.addAlphaOrbital(j2)
                dets.append(det)
        
        for k1,k2 in combinations(betaO,2):
            for l1,l2 in combinations(betaU,2):
                det = self.copy()
                det.removeBetaOrbital(k1)
                det.addBetaOrbital(l1)
                det.removeBetaOrbital(k2)
                det.addBetaOrbital(l2)
                dets.append(det)
        return dets
    
    def generateDoubleExcitationOfDet(self, numOfBasis):
        """generate all the single and double excitations of determinant in a list"""
        alphaO, betaO = self.getOrbitalIndexLists()
        alphaU, betaU = self.getUnoccupiedOrbitalsInLists(numOfBasis)
        dets = []
                
        for i in alphaO:
            for j in alphaU:
                for k in betaO:
                    for l in betaU:
                        det = self.copy()
                        det.removeAlphaOrbital(i)
                        det.addAlphaOrbital(j)
                        det.removeBetaOrbital(k)
                        det.addBetaOrbital(l)
                        dets.append(det)
        
        for i1,i2 in combinations(alphaO,2):
            for j1,j2 in combinations(alphaU,2):
                det = self.copy()
                det.removeAlphaOrbital(i1)
                det.addAlphaOrbital(j1)
                det.removeAlphaOrbital(i2)
                det.addAlphaOrbital(j2)
                dets.append(det)
        
        for k1,k2 in combinations(betaO,2):
            for l1,l2 in combinations(betaU,2):
                det = self.copy()
                det.removeBetaOrbital(k1)
                det.addBetaOrbital(l1)
                det.removeBetaOrbital(k2)
                det.addBetaOrbital(l2)
                dets.append(det)
        return dets
    
    def copy(self):
        """Return a deep copy of self"""
        return Determinant_bits(alphaObtBits=self.alphaObtBits, betaObtBits=self.betaObtBits)
    
    def __str__(self):
        a,b = self.getOrbitalIndexLists()
        return "|"+str(a)+str(b)+">"

def _test():
    det = Determinant_bits(alphaObtBits=11, betaObtBits=11)
    print det.getNumOrbitals()
    det.addAlphaOrbital(100)
    print det.getNumOrbitals()
    det.removeAlphaOrbital(100)
    print det.alphaObtBits, det.betaObtBits
    det2 = Determinant_bits(alphaObtList=[1, 4, 7],betaObtList=[2, 1, 3])
    print det.numberOfCommonOrbitals(det2)
    print det.numberOfDiffOrbitals(det2)
    print det.getOrbitalIndexLists()
    print det2.getOrbitalIndexLists()
    print det.getUniqueOrbitalsInLists(det2)
    print det.diff2OrLessOrbitals(det2)
    print det.getOrbitalPositionLists([0,3], [1])
    uniqueOrbitals = det.getUniqueOrbitalsInLists(det2)
    print det.getOrbitalPositionLists(uniqueOrbitals[0][0], uniqueOrbitals[0][1]), det2.getOrbitalPositionLists(uniqueOrbitals[1][0], uniqueOrbitals[1][1])
    print det.getSignToMoveOrbitalsToTheFront([1,3], [])
    print det.getUniqueOrbitalsInListsPlusSign(det2)
    print det.getOrbitalMixedIndexList()
    print det.getCommonOrbitalsInMixedSpinIndexList(det2)
    print det.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
    print det, det2
    det1 = Determinant_bits(alphaObtList=[0,1],betaObtList=[0,1])
    det2 = Determinant_bits(alphaObtList=[0,1],betaObtList=[0,2])
    print "Test signs", det1, det2
    print det1.getUniqueOrbitalsInListsPlusSign(det2)
    print det2.getUniqueOrbitalsInListsPlusSign(det1)
    print det2.getUnoccupiedOrbitalsInLists(5)
    for det in det1.generateSingleAndDoubleExcitationOfDet(4):
        print det

if __name__ == '__main__':
    _test()