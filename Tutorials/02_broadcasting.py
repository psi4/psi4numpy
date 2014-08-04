import numpy as np

spacer = '_' * 60

print '\nNumpy arrays have a built in "shape" which corresponds to different axes'
arr2d = np.arange(8).reshape(2,4)
print 'arr2d = '
print arr2d

print 'arr2d.shape =', arr2d.shape 

print '\nAs numpy arrays are c-contiguous the zeroth axis indicates how many rows'
print 'a numpy array has and the first axis indicates the number of columns' 

print '\nMost numpy operations have a built in "axis" argument to perform operations'
print 'on an axis'
print 'np.sum(arr2d) =', np.sum(arr2d)
print 'np.sum(arr2d, axis=0) =', np.sum(arr2d, axis=0)
print 'np.sum(arr2d, axis=1) =', np.sum(arr2d, axis=1)
 
print spacer



