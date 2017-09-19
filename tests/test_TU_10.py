from addons import *
from utils import *


tdir = 'Tutorials/10_Orbital_Optimized_Methods'


@using_numpy_113
@using_scipy
def test_10a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '10a_orbital-optimized-mp2')
