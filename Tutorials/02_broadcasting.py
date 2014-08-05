import numpy as np

spacer = '_' * 60

print '\nNumpy arrays have a built in "shape" which gives the dimensions of each axis'
arr2d = np.arange(8).reshape(2,4)
print 'arr2d = '
print arr2d

print 'arr2d.shape =', arr2d.shape 

print '\nAs numpy arrays are c-contiguous the zeroth axis indicates how many rows'
print 'a numpy array has and the first axis indicates the number of columns.' 

print '\nMost numpy operations have a built in "axis" argument to perform operations'
print 'on an axis'
print 'np.sum(arr2d) =', np.sum(arr2d)
print 'np.sum(arr2d, axis=0) =', np.sum(arr2d, axis=0)
print 'np.sum(arr2d, axis=1) =', np.sum(arr2d, axis=1)
 
print spacer

print '\nArrays can be reshaped:'
arr1 = np.arange(4)

print 'arr1 = ', arr1
print 'arr1.reshape(2, 2) = '
print arr1.reshape(2, 2)
print 'arr1.reshape(4, 1) = '
print arr1.reshape(4, 1)

print spacer

print '\nArrays of different shapes can be "broadcast" together:'

print 'arr1 + arr1.reshape(4, 1) ='
print arr1 + arr1.reshape(4,1)



