import numpy as np

spacer = '_'*60

print 'Creating a basic array "arr"'
arr = np.arange(4)
print 'arr = ', arr

print spacer

print '\nOperations by default are element-wise'
print 'arr + arr = ', arr + arr
print 'arr * arr = ', arr * arr

print spacer

print '\nThis works for arbitrary dimensional arrays'
arr2d = np.arange(16).reshape(4,4)
print 'arr2d = '
print arr2d
print '\narr2d + arr2d'
print arr2d + arr2d

print spacer

print '\nNumpy has a host of built in operations'
print 'np.sqrt(arr2d)'
print np.sqrt(arr2d)

