"""A script that runs each of the Psi4NumPy reference implementations and checks that they exit successfully."""

__author__ = "Daniel G. A. Smith"
__license__ = "BSC-3Clause"
__copyright__ = "(c) 2017, The Psi4NumPy Developers"

import glob
import subprocess
import os
import time
import shutil
import sys
import numpy as np

try:
    import psi4
except:
    raise ImportError("The Python module 'psi4' was not found!")

# How wide is the terminal
ncolumns = shutil.get_terminal_size((80, 20)).columns

# List of scripts to skip testing
exceptions = ["Coupled-Cluster/TD-CCSD.py"]

# List of folders to run the python scripts in
folders = [
"Self-Consistent-Field",
"Coupled-Cluster",
"Moller-Plesset",
"Symmetry-Adapted-Perturbation-Theory",
"Electron-Propagator"]

### Helper functions

def run_script(command):
    """Runs a shell command and returns both the success flag and output"""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    output = process.communicate()[0]
    success = process.poll() == 0
    return (success, output)

def print_banner(title):
    """Prints out a full banner across the entire screen"""

    title = " " + title + " "

    nequals = ncolumns - len(title)
    nleft = nequals // 2
   
    print(("=" * (nleft + nequals %2)) + title + ("=" * nleft)) 

def print_flag(success):
    if success:
        print("\033[92m PASSED\033[00m")
    else:
        print("\033[91m FAILED\033[00m")

### Start script

# Initial state data
print(sys.version)
print("Psi4 Version: %s" % psi4.__version__)
print("NumPy Version: %s\n" % np.version.version)

# Internal variables
failing_list = []

full_timer = time.time()
ntest = 0
nfailed = 0

for folder in folders:
    print_banner("Now testing: " + folder)
    files = glob.glob(folder + '/*.py')
    for script in files:
        if script in exceptions:
            continue

        ntest += 1
        print(script + ":", end="", flush=True)
        success, output = run_script("python " + script)
        print_flag(success)

        if not success:
            failing_list.append((script, output))
            nfailed += 1
    print("")


total_time = time.time() - full_timer

print_banner("Ran %d test in %.3f seconds" % (ntest, total_time))

print("")
if nfailed == 0:
    print("\033[92m All tests passed!\033[00m")
else:
    print("\033[91m %d tests failed!\033[00m\n" % nfailed)
    print("The following test cases failed:")
    for script, output in failing_list:
        print(script)

    print("\n")
    print_banner("Failing outputs")
    for script, output in failing_list:
        print("\nFailing output for %s" % script)
        print(output)
        print("-" * ncolumns)
    print("")




