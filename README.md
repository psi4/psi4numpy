psi4numpy
=============

#####Author: Daniel G. A. Smith
#####Contact: dsmith@auburn.edu

The overall goal of psi4numpy is to provide clear, readable code for both learning
and development. Python is used to "plug" together low level languages and
provide an interface that is both simple to use and remains relatively fast
to execute. In this case Psi4 is used for molecular properties and integrals,
numpy/scipy is used for tensor operations and linear algebra.

The input files should be run as a normal psi4 input script; however,
it should be noted the output is printed to the command line instead of merged
with the psi4 output.

If you have comments, questions, or would like to contribute to the project please
feel free to email [me](mailto:dsmith@auburn.edu).

A tutorial that covers the basics of numpy can be found [here](http://wiki.scipy.org/Tentative_NumPy_Tutorial).

#####Requirements:
- [Psi4](https://github.com/psi4/psi4public)
- [Python](python.org) 2.7+
 - [Numpy](scipy.org) 1.7.2+
 - [Scipy](numpy.scipy.org) 0.13.0+

To use python syntax highlighting in vim for `*.dat` files add the following to
your `.vimrc` file:
`au BufReadPost *.dat set syntax=python`

---

#####Update 2/19/2015
A new Psi4 has been released to the public which includes major changes to how
numpy and Psi4 data types interact.  A few of the important changes are as
follows:
- The Psi4 Matrix and Vector classes now have a `numpy_shape` attribute
  allowing arbitrary ndarray shapes to be returned. For example,
  `np.asanyarray(mints.ao_eri())` now returns a rank 4 tensor of the correct
  shape rather than a rank 2 tensor with combined indices.
- Psi4 Matix and Vector classes can now share data with a numpy ndarray. This
  is used to access base Psi4 functions such as LibFock allowing computation of
  HF based methods without exporting the N^4 ERI tensor into numpy.
- Density fitted ERI tensors and Laplace denominators are now available.
  Density fitting allows circumvention of the primary psi4numpy obstacle:
  core memory.

In addition, I am writing a major addition to the `numpy.einsum` function. The
new version will automatically factorize tensor expressions and use vendor BLAS
when available. The CCSD code runs 40x faster for small systems using the new
`numpy.einsum` when linked with vendor BLAS. More information can be found
[here](https://github.com/dgasmith/opt_einsum).

Combined with lessons learned, these changes are quite revolutionary. I will be
updating the current psi4numpy scripts over time. However, it should be noted
the new psi4numpy scripts are likely to be incompatible with older versions of
Psi4. Current psi4 scripts can be found under the branch psi4beta5.
