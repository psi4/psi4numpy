import os

from addons import *
from utils import *


tdir = 'Coupled-Cluster'

def test_CCSD_DIIS(workspace):
    script = os.getcwd() + '/../' + tdir + '/CCSD_DIIS.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_CCSD(workspace):
    script = os.getcwd() + '/../' + tdir + '/CCSD.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_CCSD_T(workspace):
    script = os.getcwd() + '/../' + tdir + '/CCSD_T.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

#def test_TD_CCSD(workspace):
#    script = os.getcwd() + '/../' + tdir + '/TD-CCSD.py'
#    with uplusx(script) as exescript:
#        workspace.run(exescript)
