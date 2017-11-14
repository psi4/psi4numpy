from addons import *
from utils import *

tdir = 'Tutorials/13_Geometry-Optimization'

def test_13a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '13a_Internal-Coordinates-Bmatrix.ipynb')

def test_13b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '13b_Hessians.ipynb')

def test_13c(workspace):
    exe_scriptified_ipynb(workspace, tdir, '13c_Hessians-updating.ipynb')

def test_13d(workspace):
    exe_scriptified_ipynb(workspace, tdir, '13d_Rational-Function-Optimization.ipynb')

def test_13e(workspace):
    exe_scriptified_ipynb(workspace, tdir, '13e_Step-Backtransformation.ipynb')

