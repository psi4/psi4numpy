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
        left_pos += (input_left.find(s), )
        right_pos += (input_right.find(s), )
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
        new_view = np.dot(
            op1.reshape(dim_left, dim_removed),
            op2.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(
            op1.reshape(dim_removed, dim_left).T,
            op2.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(
            op1.reshape(dim_left, dim_removed),
            op2.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(
            op1.reshape(dim_removed, dim_left).T,
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


class helper_diis(object):
    def __init__(self, t1, t2, max_diis):

        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()
        self.diis_vals_t1 = [t1.copy()]
        self.diis_vals_t2 = [t2.copy()]
        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis

    def add_error_vector(self, t1, t2):

        # Add DIIS vectors
        self.diis_vals_t1.append(t1.copy())
        self.diis_vals_t2.append(t2.copy())
        # Add new error vectors
        error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
        error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
        self.diis_errors.append(np.concatenate((error_t1, error_t2)))
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

    def extrapolate(self, t1, t2):

        # Limit size of DIIS vector
        if (len(self.diis_vals_t1) > self.max_diis):
            del self.diis_vals_t1[0]
            del self.diis_vals_t2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_vals_t1) - 1

        # Build error matrix B
        B = np.ones((self.diis_size + 1, self.diis_size + 1)) * -1
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = np.dot(e1, e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 >= n2: continue
                B[n1, n2] = np.dot(e1, e2)
                B[n2, n1] = B[n1, n2]

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        # Build residual vector
        resid = np.zeros(self.diis_size + 1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.linalg.solve(B, resid)

        # Calculate new amplitudes
        t1 = np.zeros_like(self.oldt1)
        t2 = np.zeros_like(self.oldt2)
        for num in range(self.diis_size):
            t1 += ci[num] * self.diis_vals_t1[num + 1]
            t2 += ci[num] * self.diis_vals_t2[num + 1]

        # Save extrapolated amplitudes to old_t amplitudes
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

        return t1, t2
