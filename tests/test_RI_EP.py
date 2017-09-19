from addons import *
from utils import *


tdir = 'Electron-Propagator'


def test_EP2(workspace):
    exe_py(workspace, tdir, 'EP2')


def test_EP2_SO(workspace):
    exe_py(workspace, tdir, 'EP2_SO')


def test_EP3_SO(workspace):
    exe_py(workspace, tdir, 'EP3_SO')
