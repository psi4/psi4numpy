import os

from addons import *
from utils import *


tdir = 'Electron-Propagator'

def test_EP2(workspace):
    script = os.getcwd() + '/../' + tdir + '/EP2.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_EP2_SO(workspace):
    script = os.getcwd() + '/../' + tdir + '/EP2_SO.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_EP3_SO(workspace):
    script = os.getcwd() + '/../' + tdir + '/EP3_SO.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)
