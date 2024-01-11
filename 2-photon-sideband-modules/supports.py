# -*- coding: utf-8 -*-
import numpy as np
import time, sys
from IPython.display import clear_output
from qutip import *


pi = np.pi
exp = np.exp
sqrt = np.sqrt
hbar = 1.0546e-34*1e-9  # reduced Planck constant
All = 'all'


psi_0pi = lambda Nq : basis(Nq, 0)
psi_0 = psi_0pi
psi_halfpi = lambda Nq : (basis(Nq, 0) + basis(Nq, 1)).unit()
psi_pi = lambda Nq : basis(Nq, 1)
psi_1pi = psi_pi
psi_2pi = psi_0pi


def ops(*args):
    """
    Returns system operators based on input arguments. The order
    of the input arguments defines the convention of order of the
    elements of the system (for example qubits and cavities).
    
    Input
    -----
    *args : int
        Every argument represents the number of levels of a system element
    
    Returns
    -------
    a# : qutip.Qobj class object
        Lowering operator for element #
    n# : qutip.Qobj class object
        Number operator for element #
    """
    if len(args) == 1:
        a0 = destroy(args[0])
        n0 = a0.dag()*a0
        return a0, n0
    
    elif len(args) == 2:
        a0 = tensor(destroy(args[0]), qeye(args[1]))
        n0 = a0.dag()*a0
        a1 = tensor(qeye(args[0]), destroy(args[1]))
        n1 = a1.dag()*a1
        return a0, a1, n0, n1
    
    elif len(args) == 3:
        a0 = tensor(destroy(args[0]), qeye(args[1]), qeye(args[2]))
        n0 = a0.dag()*a0
        a1 = tensor(qeye(args[0]), destroy(args[1]), qeye(args[2]))
        n1 = a1.dag()*a1
        a2 = tensor(qeye(args[0]), qeye(args[1]), destroy(args[2]))
        n2 = a2.dag()*a2
        return a0, a1, a2, n0, n1, n2
    
    elif len(args) == 4:
        a0 = tensor(destroy(args[0]), qeye(args[1]), qeye(args[2]), qeye(args[3]))
        n0 = a0.dag()*a0
        a1 = tensor(qeye(args[0]), destroy(args[1]), qeye(args[2]), qeye(args[3]))
        n1 = a1.dag()*a1
        a2 = tensor(qeye(args[0]), qeye(args[1]), destroy(args[2]), qeye(args[3]))
        n2 = a2.dag()*a2
        a3 = tensor(qeye(args[0]), qeye(args[1]), qeye(args[2]), destroy(args[3]))
        n3 = a3.dag()*a3
        return a0, a1, a2, a3, n0, n1, n2, n3


def update_progress(prog, length=50):
    """
    Displays progress bar.
    
    prog : float
        Fraction of total calculation completed.
    """
    if isinstance(prog, int):
        prog = float(prog)
    if not isinstance(prog, float):
        prog = 0
    if prog < 0:
        prog = 0
    if prog >= 1:
        prog = 1
    block = int(round(length * prog))
    clear_output(wait = True)
    print("Progress: |{0}| {1:.1f}%".format( "█" * block + " " * (length - block), prog * 100))
