from addons import *
from utils import *
import pytest


tdir = 'Coupled-Electron-Pair-Approximation'

# The below tests rely on the CURRENT CORRELATION ENERGY variable, which isn't set in
# the version of Psi4Numpy used in the current test suite. Remove xfail upon Psi update.

@pytest.mark.xfail
def test_LCCD(workspace):
    exe_py(workspace, tdir, 'LCCD')

@pytest.mark.xfail
def test_LCCSD(workspace):
    exe_py(workspace, tdir, 'LCCSD')

@pytest.mark.xfail
def test_OLCCD(workspace):
    exe_py(workspace, tdir, 'OLCCD')

def test_DFLCCD(workspace):
    exe_py(workspace, tdir, 'DF-LCCD')

def test_DFLCCSD(workspace):
    exe_py(workspace, tdir, 'DF-LCCSD')

