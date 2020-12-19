from addons import *
from utils import *


tdir = 'One-Electron-Property'

def test_ir_ints(workspace):
    exe_py(workspace, tdir, 'IR-Intensities/intensities')

def test_example(workspace):
    exe_py(workspace, tdir, 'Restrained-Electrostatic-Potential/example')

def test_example2(workspace):
    exe_py(workspace, tdir, 'Restrained-Electrostatic-Potential/example2')
