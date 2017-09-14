import os

from addons import *
from utils import *


tdir = 'Moller-Plesset'

def test_DF_MP2(workspace):
    script = os.getcwd() + '/../' + tdir + '/DF-MP2.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_MP2(workspace):
    script = os.getcwd() + '/../' + tdir + '/MP2.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_MP3(workspace):
    script = os.getcwd() + '/../' + tdir + '/MP3.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_MP3_SO(workspace):
    script = os.getcwd() + '/../' + tdir + '/MP3-SO.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_MPn(workspace):
    script = os.getcwd() + '/../' + tdir + '/MPn.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)
