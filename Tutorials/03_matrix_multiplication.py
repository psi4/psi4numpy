import numpy as np

spacer = '_' * 60

print 'There is many ways to complete matrix multiplication in numpy:'

print '\nFirst lets build two 2d arrays:'
arr1 = np.arange(4).reshape(2, 2)
arr2 = np.arange(4).reshape(2, 2) + 2
print 'arr1 =', arr1
print 'arr2 =', arr2

print '\nUsing np.dot, this will also call numpys linked BLAS:'
print 'np.dot(arr1, arr2) ='
print np.dot(arr1, arr2)

print '\nUsing broadcasting:'
print 'np.sum(arr1[..., None] * arr2, axis=1) ='
print np.sum(arr1[..., None] * arr2, axis=1)

print '\nUsing einsum:'
print "np.einsum('ik,kj->ij', arr1, arr2) = "
print np.einsum('ik,kj->ij', arr1, arr2)

print spacer

print '\nThere is also the matrix class which is a wrapper on numpy arrays.'
print 'In this case the * operator is overloaded for matrix multiplication.'

print '\nmat1 = np.matrix(arr1)'
print 'mat2 = np.matrix(arr2)'
mat1 = np.matrix(arr1)
mat2 = np.matrix(arr2)

print 'mat1 * mat2 =' 
print mat1 * mat2 
