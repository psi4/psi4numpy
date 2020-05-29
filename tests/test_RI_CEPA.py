from addons import *
from utils import *


tdir = 'Coupled-Electron-Pair-Approximation'


def test_LCCD(workspace):
    exe_py(workspace, tdir, 'LCCD')

def test_LCCSD(workspace):
    exe_py(workspace, tdir, 'LCCSD')

def test_OLCCD(workspace):
    exe_py(workspace, tdir, 'OLCCD')

