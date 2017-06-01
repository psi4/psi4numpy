

import numpy as np

a = np.random.random((4, 4))
a += np.eye(4)

print np.linalg.eigh(a)[1]

mat = Matrix.from_array(a)
print np.array(mat.partial_cholesky_factorize(1.e-12, False))



