from addons import *
from utils import *


tdir = 'Tutorials/12_MD'

@using_matplotlib
def test_12a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '12a_basics')

@using_matplotlib
@using_scipy
def test_12b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '12b_ewald')

