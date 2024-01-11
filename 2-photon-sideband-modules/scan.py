import time
import numpy as np
import matplotlib.pyplot as plt
from plots import *
from supports import *
from process import *
from calculate import *
from envelopes import *
from qutip import *
from ipywidgets import widgets
from IPython.display import display


def sbsample(Nq, omega_q, omega_c, Ec, g, omega_dsi, sb, Nt, H, H_args, convergent, refinement, psi0, c_ops, Np_per_batch,
             options, home, parallel, *args, **kwargs):
    """
    Performs a single- or double-tone sideband transition simulation
    with the given input parameters. Plots the expectation
    values of the qubit & cavity, and the combined probability
    |e0>-|g1> in the case of red sideband transitions, or
    |e1>-|g0> in the case of blue sideband transitions.
    
    Due to the use of the pool.starmap function, the additional
    arguments of *args, dependent on Nt, have to be passed in
    a definite order.
    Nt = 1 : Omega_d
    Nt = 2 : Omega_dq, Omega_dc, d_omega
    
    Input
    -----
    The input parameters are equal to the names in 2p_sideband.ipynb.
    
    Returns
    -------
    figqc : matplotlib.pyplot.Figure class object
        Figure with expected qubit and cavity occupation number
    fig : matplot.pyplot.Figure class object
        Figure with combined probabilities
    Swept_freq : float. 
        The swept frequency.
    fidelity : float.
        Fidelity of sideband transition.
    """
    from envelopes import drive
    
    i = omega_dsi[0]
    omega_swept = omega_dsi[1]
    
    Nc = 10  # number of levels in resonator 1
    
    Np = 100*int(H_args['t3'])  # number of discrete time steps for which to store the output
    b, a, nq, nc = ops(Nq, Nc)  # Operators
    if Nt == 1:
        omega_d = omega_swept
        H_args['omega_d'] = omega_d
    elif Nt == 2:
        d_omega = args[2]
        omega_dq = omega_swept
        omega_dc =  omega_c - d_omega
        H_args['omega_dq'] = omega_dq
        H_args['omega_dc'] = omega_dc
    e_ops = [nq, nc]

    srcfolder = calculate(H, psi0, e_ops, c_ops, H_args, convergent, refinement, options, Nc, Np, Np_per_batch,
                          home, parallel, verbose=False, method='me')
    
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    if i < 10:
        num = "0" + str(i)
    elif i >= 10:
        num = str(i)
    
    print(" ")
    
    if Nt == 1:
        Omega_d = args[0]
        Swept_freq = omega_swept/2/pi
        print("omega_d = {}".format(omega_d/2/pi))
        if sb == 'red':
            fidelity = abs(max(e0-g1)-min(e0-g1))/2
            print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
            print("fidelity = {}".format(round(fidelity),4))
            expect_title = "omega_d = {}".format(omega_d/2/pi)
            cp_title = "omega_d = {}, min = {}, max = {}".format(omega_d/2/pi, round(min(e0-g1), 4), round(max(e0-g1), 4))
            figqc, axqc = sb_expect(times, expect, sb, Nt, H_args, convergent, coupling, xlim=None, ylim=None, figsize=[15,3],
                              omega_d=omega_d, wsb=0, title=expect_title, Omega_d=Omega_d)
            fig, axp = sb_combined_probs(times, sb, Nt, H_args, convergent, coupling,
                                    e0=e0, g1=g1, omega_d=omega_d, wsb=0, title=cp_title, Omega_d=Omega_d)
        elif sb == 'blue':
            fidelity = abs(max(e1-g0)-min(e1-g0))/2
            print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
            print("fidelity = {}".format(round(fidelity),4))
            expect_title = "omega_d = {}".format(omega_d/2/pi)
            cp_title = "omega_d = {}, min = {}, max = {}".format(omega_d/2/pi, round(min(e1-g0), 4), round(max(e1-g0), 4))
            figqc, axqc = sb_expect(times, expect, sb, Nt, H_args, convergent, coupling, xlim=None, ylim=None, figsize=[15,3],
                              omega_d=omega_d, wsb=0, title=expect_title, Omega_d=Omega_d)
            fig, axp = sb_combined_probs(times, sb, Nt, H_args, convergent, coupling,
                                    e1=e1, g0=g0, omega_d=omega_d, wsb=0, title=cp_title, Omega_d=Omega_d)
        fig.savefig(home + "temp/fig{}_{}.png".format(num, omega_d/2/pi))
        figqc.savefig(home + "temp/figqc{}_{}.png".format(num, omega_d/2/pi))
    
    elif Nt == 2:
        Omega_dq = args[0]
        Omega_dc = args[1]
        Swept_freq = omega_swept/2/pi
        print("omega_dq = {}".format(omega_dq/2/pi))
        if sb == 'red':
            fidelity = abs(max(e0-g1)-min(e0-g1))/2
            print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
            print("fidelity = {}".format(round(fidelity),4))
            expect_title = "omega_dq = {}".format(omega_dq/2/pi)
            cp_title = "omega_dq = {}, min = {}, max = {}".format(omega_dq/2/pi, round(min(e0-g1), 4), round(max(e0-g1), 4))
            figqc, axqc = sb_expect(times, expect, sb, Nt, H_args, convergent, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wsb=0, title=expect_title, Omega_dq=Omega_dq, Omega_dc=Omega_dc)
            fig, axp = sb_combined_probs(times, sb, Nt, H_args, convergent, coupling,
                                    xlim=None, ylim=None, figsize=[15,3], e0=e0, g1=g1, wsb=0,
                                    title=cp_title, Omega_dq=Omega_dq, Omega_dc=Omega_dc)
        elif sb == 'blue':
            fidelity = abs(max(e1-g0)-min(e1-g0))/2
            print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
            print("fidelity = {}".format(round(fidelity),4))
            expect_title = "omega_dq = {}".format(omega_dq/2/pi)
            cp_title = "omega_dq = {}, min = {}, max = {}".format(omega_dq/2/pi, round(min(e1-g0), 4), round(max(e1-g0), 4))
            figqc, axqc = sb_expect(times, expect, sb, Nt, H_args, convergent, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wsb=0, title=expect_title, Omega_dq=Omega_dq, Omega_dc=Omega_dc)
            fig, axp = sb_combined_probs(times, sb, Nt, H_args, convergent, coupling,
                                    xlim=None, ylim=None, figsize=[15,3], e1=e1, g0=g0, wsb=0,
                                    title=cp_title, Omega_dq=Omega_dq, Omega_dc=Omega_dc)
        fig.savefig(home + "temp/fig{}_{}.png".format(num, omega_dq/2/pi))
        figqc.savefig(home + "temp/figqc{}_{}.png".format(num, omega_dq/2/pi))
    plt.close(fig)
    plt.close(figqc)
    return figqc, fig, Swept_freq, fidelity


def sbsample_visualize_sweep(Nq, omega_q, omega_c, Ec, g, omega_d, sb, Nt, H, H_args, psi0, Np_per_batch,
             options, home, parallel, *args):
    from envelopes import drive
    
    i = omega_d[0]
    omega_d = omega_d[1]
    
    Nc = 10  # number of levels in resonator 1
    
    Np = 100*int(H_args['t3'])  # number of discrete time steps for which to store the output
    b, a, nq, nc = ops(Nq, Nc)  # Operators
    if Nt == 1:
        H_args['omega_d'] = omega_d
    elif Nt == 2:
        d_omega = args[2]
        omega_dq = omega_d
        omega_dc =  omega_c - d_omega
        H_args['omega_dq'] = omega_dq
        H_args['omega_dc'] = omega_dc
    e_ops = [nq, nc]
    c_ops = []
        
    srcfolder = calculate(H, psi0, e_ops, c_ops, H_args, options, Nc, Np, Np_per_batch,
                          home, parallel, verbose=False)
    
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    if i < 10:
        num = "0" + str(i)
    elif i >= 10:
        num = str(i)
    
    print(" ")
    
    if Nt == 1:
        Omega_d = args[0]
        print("$\\omega_d /2\\pi = ${} GHz".format(omega_d/2/pi))
        if sb == 'red':
            expect_title = "$\\omega_d /2\\pi = ${} GHz".format(np.round(omega_d/2/pi, 4))
            cp_title = "$\\omega_d /2\\pi = ${} GHz".format(np.round(omega_d/2/pi, 4))
            figqc, axqc = sb_expect_temporary(times, expect, sb, Nt, H_args, coupling, xlim=[0,1000], ylim=[-0.02, 1.02], figsize=[12,4],
                              omega_d=omega_d, wsb=0, title=expect_title, Omega_d=Omega_d)
            fig, axp = sb_combined_probs_temporary(times, sb, Nt, H_args, coupling, figsize=[12,4],
                                    e0=e0, g1=g1, omega_d=omega_d, wsb=0, xlim=[0,1000], ylim=[-1.02, 1.02], title=cp_title, Omega_d=Omega_d)
        elif sb == 'blue':
            expect_title = "$\\omega_d /2\\pi = ${} GHz".format(np.round(omega_d/2/pi, 4))
            cp_title = "$\\omega_d /2\\pi = ${} GHz".format(np.round(omega_d/2/pi, 4))
            figqc, axqc = sb_expect_temporary(times, expect, sb, Nt, H_args, coupling, xlim=[0,1000], ylim=[-0.02, 1.02], figsize=[12,4],
                              omega_d=omega_d, wsb=0, title=expect_title, Omega_d=Omega_d)
            fig, axp = sb_combined_probs_temporary(times, sb, Nt, H_args, coupling, figsize=[12,4],
                                    e1=e1, g0=g0, omega_d=omega_d, wsb=0, xlim=[0,1000], ylim=[-1.02, 1.02], title=cp_title, Omega_d=Omega_d)
    
    elif Nt == 2:
        Omega_dq = args[0]
        Omega_dc = args[1]
        print("$\\omega_dq /2\\pi = ${} GHz".format(np.round(omega_dq/2/pi, 4)))
        if sb == 'red':
            expect_title = "$\\omega_dq /2\\pi = ${} GHz".format(np.round(omega_dq/2/pi, 4))
            cp_title = "$\\omega_dq /2\\pi = ${} GHz".format(np.round(omega_dq/2/pi, 4))
            figqc, axqc = sb_expect_temporary(times, expect, sb, Nt, H_args, coupling, xlim=[0,1000], ylim=[-0.02, 1.02], figsize=[12,4],
                              wsb=0, title=expect_title, Omega_dq=Omega_dq, Omega_dc=Omega_dc)
            fig, axp = sb_combined_probs_temporary(times, sb, Nt, H_args, coupling,
                                   xlim=[0,1000], ylim=[-1.02, 1.02], figsize=[15,3], e0=e0, g1=g1, wsb=0,
                                    title=cp_title, Omega_dq=Omega_dq, Omega_dc=Omega_dc)
        elif sb == 'blue':
            expect_title = "$\\omega_dq /2\\pi = ${} GHz".format(np.round(omega_dq/2/pi, 4))
            cp_title = "$\\omega_dq /2\\pi = ${} GHz".format(np.round(omega_dq/2/pi, 4))
            figqc, axqc = sb_expect_temporary(times, expect, sb, Nt, H_args, coupling, xlim=[0,1000], ylim=[-0.02, 1.02], figsize=[12,4],
                              wsb=0, title=expect_title, Omega_dq=Omega_dq, Omega_dc=Omega_dc)
            fig, axp = sb_combined_probs_temporary(times, sb, Nt, H_args, coupling, 
                                    xlim=[0,1000], ylim=[-1.02, 1.02], figsize=[12,4], e1=e1, g0=g0, wsb=0,
                                    title=cp_title, Omega_dq=Omega_dq, Omega_dc=Omega_dc)
        fig.savefig(home + "temp/fig{}_{}.png".format(num, omega_dq/2/pi))
        figqc.savefig(home + "temp/figqc{}_{}.png".format(num, omega_dq/2/pi))
    plt.close(fig)
    plt.close(figqc)
    return figqc, fig


def qfs(Nq, omega_q, Ec, omega_p, H, H_args, psi0, Nc, Np, Np_per_batch, options, home, parallel):
    """
    Applies a probe tone to an uncoupled qubit to find its transition frequency,
    which can be shifted due to a dispersive drive.
    
    Input
    -----
    The input parameters are equal to the names in 2p_sideband.ipynb.
    """
    i = omega_p[0]
    omega_p = omega_p[1]
    H_args['omega_p'] = omega_p
    b, nq = ops(Nq)
    e_ops = [nq]
    c_ops = []
        
    srcfolder = calculate(H, psi0, e_ops, c_ops, H_args, options, Nc, Np, Np_per_batch,
                          home, parallel, verbose=False)
    quants = ['times', 'expect']
    combine_batches(srcfolder, quants=quants, return_data=False)
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    if i < 10:
        num = "0" + str(i)
    elif i >= 10:
        num = str(i)
    
    print("\nshift  =", (omega_p-omega_q)/2/pi)
    print("omega_p     =", omega_p/2/pi)
    print("max    =", max(expect[0]))
    
    plt.figure(figsize=[15,3])
    plt.plot(times, expect[0], c='k')
    plt.title("shift {}, omega_p {}, max {}".format(np.round((omega_p-omega_q)/2/pi,4), np.round(omega_p/2/pi,4), np.round(max(expect[0]),4)))
    plt.savefig(home + "temp/fig{}_{}.png".format(num, (omega_p-omega_q)/2/pi))


def cfs(Nq, Nc, omega_c, Ec, omega_p, H, H_args, psi0, Np_per_batch, options, home, parallel):
    i = omega_p[0]
    omega_p = omega_p[1]
    H_args['omega_p'] = omega_p
    Np = 100*int(H_args['t3'])     # number of discrete time steps for which to store the output
    b, a, nq, nc = ops(Nq, Nc)  # Operators
    e_ops = [nq, nc]
    c_ops = []
    
    srcfolder = calculate(H, psi0, e_ops, c_ops, H_args, options, Nc, Np, Np_per_batch,
                          home, parallel, verbose=False)
    quants = ['times', 'expect']
    combine_batches(srcfolder, quants=quants, return_data=False)
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    if i < 10:
        num = "0" + str(i)
    elif i >= 10:
        num = str(i)
    
    print("omega_p     =", omega_p/2/pi)
    print("max    =", max(expect[1]))
    
    plt.figure(figsize=[15,3])
#    plt.plot(times, expect[0], c='b')
    plt.plot(times, expect[1], c='r')
    plt.title("shift {}, omega_p {}, max {}".format(np.round((omega_p-omega_c)/2/pi,4), np.round(omega_p/2/pi,4), np.round(max(expect[1]),6)))
    plt.savefig(home + "temp/fig{}_{}.png".format(num, (omega_p-omega_c)/2/pi))
