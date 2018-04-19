from addons import *
from utils import *


tdir = 'One-Electron-Property/Restrained-Electrostatic-Potential'

def test_example(workspace):
    exe_py(workspace, tdir,'example')

def test_example2(workspace):
    exe_py(workspace, tdir,'example2')
