import os

from addons import *
from utils import *


tdir = 'Configuration-Interaction'

def test_CI_DL(workspace):
    exe_py(workspace, tdir, 'CI_DL')
    #script = os.getcwd() + '/../' + tdir + '/CI_DL.py'
    #with uplusx(script) as exescript:
    #    workspace.run(exescript)

@using_scipy
def test_CISD(workspace):
    exe_py(workspace, tdir, 'CISD')
    #script = os.getcwd() + '/../' + tdir + '/CISD.py'
    #with uplusx(script) as exescript:
    #    workspace.run(exescript)

def test_CIS(workspace):
    exe_py(workspace, tdir, 'CIS')
    #script = os.getcwd() + '/../' + tdir + '/CIS.py'
    #with uplusx(script) as exescript:
    #    workspace.run(exescript)

@using_scipy
def test_FCI(workspace):
    exe_py(workspace, tdir, 'FCI')
    #script = os.getcwd() + '/../' + tdir + '/FCI.py'
    #with uplusx(script) as exescript:
    #    workspace.run(exescript)

