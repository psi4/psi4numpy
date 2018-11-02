from addons import *
from utils import *


tdir = 'Moller-Plesset'


def test_DF_MP2(workspace):
    exe_py(workspace, tdir, 'DF-MP2')


def test_MP2(workspace):
    exe_py(workspace, tdir, 'MP2')


def test_sDF_MP2(workspace):
    exe_py(workspace, tdir, 'sDF-MP2')


def test_MP3(workspace):
    exe_py(workspace, tdir, 'MP3')


def test_MP3_SO(workspace):
    exe_py(workspace, tdir, 'MP3-SO')


def test_MPn(workspace):
    exe_py(workspace, tdir, 'MPn')


def test_MP2_Gradient(workspace):
    exe_py(workspace, tdir, 'MP2_Gradient')
