from addons import *
from utils import *


tdir = 'Coupled-Electron-Pair-Approximation'


def test_LCCD(workspace):
    exe_py(workspace, tdir, 'LCCD')

