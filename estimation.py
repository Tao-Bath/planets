#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESTIMATE PERMEABILITY TENSOR K

Created on Wed Apr  8 20:24:46 2020

@author: taozhang
"""

from fenics import *
from fenics_adjoint import *
from numpy import reshape, mean
import matplotlib.pyplot as plt
import time

# function: compute pressure (default: the function is NOT taped by adjoint)
def solve_pressure(bc, K, f, V, annotate=False):
    # Variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a1 = dot(dot(K,grad(u)), grad(v))*dx
    L1 = f*v*dx
    # Compute solution
    u = Function(V, annotate=annotate)
    solve(a1 == L1, u, bc, annotate=annotate)
    return u

# function: compute concentration (generate synthetic data)
def solve_concent(bc, mesh, V, K, f, q, u, D_coef, dt, num_steps):
    c = TrialFunction(V)
    g = TestFunction(V)
    c_0 = Function(V, annotate=False)
    ts = Constant(dt) # use constant instead of numerical in variational form
    def velocity(u):  # velocity
        return -dot(K,grad(u))
    # Residual
    r = (c - c_0)/ts + dot(velocity(u),grad(c)) - div(dot(D_coef*K, grad(c))) - q
    # Variational form
    F = ((c - c_0)/ts*g + dot(velocity(u),grad(c))*g + dot(dot(D_coef*K, grad(c)),grad(g)) - q*g)*dx
    # Add SUPG stabilisation terms
    vnorm = sqrt(dot(velocity(u), velocity(u)))
    h = CellDiameter(mesh)
    delta = h/(2.0*vnorm)
    # delta = 0.5*h*pow(4.0/(Pe*h)+2.0*vnorm,-1.0)
    F += delta*dot(velocity(u), grad(g)) * r * dx

    a2, L2 = lhs(F), rhs(F)
    c = Function(V, annotate=False)
    t = 0
    # vtkfile_d = File('d.pvd')
    xdmffile_d = XDMFFile('synConc.xdmf')
    hdf = HDF5File(mesh.mpi_comm(), "synData.h5", "w")
    # timeseries_d = TimeSeries('conc_series')

    for n in range(num_steps):
        # Update current time
        t += dt
        # Compute solution
        solve(a2 == L2, c, bc, annotate=False)
        # save solution
        # vtkfile_d << (c, t)
        xdmffile_d.write(c, t)
        hdf.write(c, "d", n)
        # timeseries_d.store(c.vector(), t)
        # Update previous solution c0
        c_0.assign(c, annotate=False)
    xdmffile_d.close()
    hdf.close()
    # del hdf
    return 

# function: compute concentration and return functional j
def solve_concent_adj(bc, mesh, V, u, f, q, K, D_coef, dt, num_steps, synConc, writeFlag, path):   
    c = TrialFunction(V)
    g = TestFunction(V)
    c_0 = Function(V)
    d = Function(V, name="data")
    ts = Constant(dt)
    def velocity(u):  # velocity
        return -dot(K,grad(u))
    # Residual
    r = (c - c_0)/ts + dot(velocity(u),grad(c)) - div(dot(D_coef*K, grad(c))) - q
    # Variational form
    F = ((c - c_0)/ts*g + dot(velocity(u),grad(c))*g + dot(dot(D_coef*K, grad(c)),grad(g)) - q*g)*dx
    # Add SUPG stabilisation terms
    vnorm = sqrt(dot(velocity(u), velocity(u)))
    h = CellDiameter(mesh)
    delta = h/(2.0*vnorm)
    # delta = 0.5*h*pow(4.0/(Pe*h)+2.0*vnorm,-1.0)
    F += delta*dot(velocity(u), grad(g)) * r * dx
    
    a2, L2 = lhs(F), rhs(F)
    A2 = assemble(a2)   # lhs will not change, only need to assemble once
    bc.apply(A2)
    c = Function(V)
    t = 0
    j = 0.5*float(dt)*assemble((c - d)**2*dx)
    
    if writeFlag == True:
        xdmffile_c = XDMFFile(path + 'simConc.xdmf')

    for n in range(num_steps):
        # Update current time
        t += dt
        # Compute solution
        b2 = assemble(L2) # assemble rhs
        bc.apply(b2)
        solve(A2, c.vector(), b2)
        # solve(a2 == L2, c)
        # save
        if writeFlag == True:
            xdmffile_c.write(c, t)
        # Update data function
        d.assign(synConc[t])
    
        # Implement a trapezoidal rule
        if n < num_steps-1: # n starts from 0
            weight = 1
        else:
            weight = 0.5    # last slice

        j += weight*float(dt)*assemble((c - d)**2*dx)
        # Update previous solution
        c_0.assign(c)
    if writeFlag == True:
        xdmffile_c.close()
    return j

# function: read synthetic data from file
def read_synData(mesh, dt, num_steps, V, slicing):
    # read pressure
    u_syn = Function(V, name="tmp", annotate=False)
    hdf = HDF5File(mesh.mpi_comm(), "syn_ps.h5", "r")
    hdf.read(u_syn, "ps")
    hdf.close()
    # read concentration
    from collections import OrderedDict
    hdf = HDF5File(mesh.mpi_comm(), "synData.h5", "r")
    # timeseries_d = TimeSeries('conc_series')
    synConc = OrderedDict()
    d = Function(V, name="tmp", annotate=False)
    t = 0
    xdmffile_obs = XDMFFile('obsConc.xdmf')
    
    for n in range(0, num_steps, int(1/slicing)):
        n += int(1/slicing) - 1  # pick corresponding slice
        t += dt
        synConc[t] = Function(V, annotate=False)
        dataset = "d/vector_%d"%n
        hdf.read(d, dataset)
        xdmffile_obs.write(d, t)
        # timeseries_d.retrieve(d.vector(), t)
        synConc[t].assign(d, annotate=False)
    hdf.close()
    xdmffile_obs.close()
    return u_syn, synConc

# function: generate and save synthetic data    
def generate_synData(bc1, bc2, K, f, V, VL, mesh, q, D_coef, dt, num_steps):
    # plot exact perm K
    K_mean = [assemble(K[0,0]*dx), assemble(K[0,1]*dx), assemble(K[1,0]*dx), assemble(K[1,1]*dx)]
    plot_K(K, K_mean, '', 'K_exa.eps')
    # compute pressure
    u = solve_pressure(bc1, K, f, V)
    # plot pressure and velocity
    plot_p_v(u, VL, K, 'syn_')
    # save pressure
    hdf = HDF5File(mesh.mpi_comm(), "syn_ps.h5", "w")
    hdf.write(u, "ps")
    hdf.close()
    # compute and save concentration
    solve_concent(bc2, mesh, V, K, f, q, u, D_coef, dt, num_steps)
    return

#function: estimate permeability tensor
def estimate_K(mesh, V, VL, K, K_exa, bc1, bc2, f, q, D_coef, dt, num_steps, u_syn, synConc, alpha, beta, path):
    tic = time.time()  # start recording time
    # start estimating
    # compute pressure
    u = solve_pressure(bc1, K, f, V, True)
    # compute concentration and return functional j
    writeFlag = False  # do not save concentration until the last run
    j = solve_concent_adj(bc2, mesh, V, u, f, q, K, D_coef, dt, num_steps, synConc, writeFlag, path)
    # control variable K
    m = Control(K)
    J = alpha * j+ beta * assemble(inner(grad(K), grad(K))*dx)  # minimise grad K

    # callback for conv of functional value and gradient executed in ReducedFunctional loop
    fun_val = []
    grad_val = []
    def derivative_cb(j,dj,m):
        fun_val.append(j)
        grad_val.append(max(abs(dj.vector().get_local())))

    reduced_functional = ReducedFunctional(J, m, derivative_cb_post = derivative_cb)
    
    # callback for convergence of K executed in minimize loop
    K11 = []; K12 = []; K21 = []; K22 = []
    def iter_cb(m):
        m = reshape(m, (3, mesh.num_vertices()),order='F')
        Kmean = mean(m, axis=1)
        K11.append(Kmean[0])
        K12.append(Kmean[1])
        K21.append(Kmean[1])
        K22.append(Kmean[2])

    K_opt = minimize(reduced_functional, method = "L-BFGS-B", bounds = (-0.0001, 1.0001), options = {"disp": True, "maxiter": 500, "gtol": 1e-9, "ftol": 1e-9}, callback = iter_cb)
    K.assign(K_opt)  # update optimised perm

    # compute simulated pressure
    u_sim = solve_pressure(bc1, K, f, V)
    # plot simulated pressure and velocity
    plot_p_v(u_sim, VL, K, 'sim_', path)
    # compute simulated concentration and return final functional j vale (misfit of sim and syn data)
    writeFlag = True
    j = solve_concent_adj(bc2, mesh, V, u_sim, f, q, K, D_coef, dt, num_steps, synConc, writeFlag, path)

    # compute and store error in txt file
    txt = open(path + 'print.txt','w')
    print("Error in control:  %e." % errornorm(K, K_exa), file=txt)
    print("Error in pressure: %e." % errornorm(u_sim, u_syn), file=txt)
    print("Error in state:    %e." % sqrt(abs(j)), file=txt)
    print("--- Num of iterations in optim: %d  ---" % len(K11), file=txt)
    print("--- Elapsed time: %s seconds ---" % (time.time() - tic), file=txt)
    txt.close()

    # compute mean value of K component (unit area)
    K_mean = [assemble(K[0,0]*dx), assemble(K[0,1]*dx), assemble(K[1,0]*dx), assemble(K[1,1]*dx)]

    # plot optimaised K components and percent error
    plot_K(K, K_mean, path)
    plot_K_error_per(K, K_exa, path)
    # plot convergence of K
    plot_K_conv(K11, K12, K21, K22, [assemble(K_exa[0,0]*dx), assemble(K_exa[0,1]*dx), assemble(K_exa[1,0]*dx), assemble(K_exa[1,1]*dx)], path)
    # plot functional value and gradient
    plot_fval(fun_val, path)
    plot_grad(grad_val, path) 
    # save K
    hdf = HDF5File(mesh.mpi_comm(), path + 'K.h5', "w")
    hdf.write(K, "K")
    hdf.close()
    return K

############ post-processing ############
# plot pressure and velocity
def plot_p_v(u, VL, K, flag, path=''):
    # plot pressure
    plt.figure()
    p1 = plot(u, title='Pressure and velocity')
    # p1.set_clim(0, 4)
    plt.colorbar(p1)
    # plot velocity
    w1 = project(-dot(K, grad(u)), VL, annotate=False)
    plot(w1)
    plt.savefig(path + flag + 'ps_vl.eps')
  
# plot permeability tensor K (2 x 2 x number of nodes)    
def plot_K(K, K_mean, path='', filename='K.eps'):
    plt.figure(figsize=(10,7.5))    
    plt.subplot(221)
    p11 = plot(K[0,0], title='$K^{(1,1)}_{mean}=%.2f$' %K_mean[0])
    plt.colorbar(p11)
    
    plt.subplot(222)
    p12 = plot(K[0,1], title='$K^{(1,2)}_{mean}=%.2f$' %K_mean[1])
    plt.colorbar(p12)
    
    plt.subplot(223)
    p13 = plot(K[1,0], title='$K^{(2,1)}_{mean}=%.2f$' %K_mean[2])
    plt.colorbar(p13)
    
    plt.subplot(224)
    p14 = plot(K[1,1], title='$K^{(2,2)}_{mean}=%.2f$' %K_mean[3])
    plt.colorbar(p14)
    plt.suptitle('Permeability tensor')
    
    plt.savefig(path + filename)
    
# plot the percentage of error of K
def plot_K_error_per(K, K_exa, path):
    plt.figure(figsize=(10,7.5))    
    plt.subplot(221)
    p11 = plot(abs(K[0,0]-K_exa[0,0])*2*100/(abs(K_exa[0,0])+abs(K[0,0])), title='K11')
    plt.colorbar(p11)
    
    plt.subplot(222)
    p12 = plot(abs(K[0,1]-K_exa[0,1])*2*100/(abs(K_exa[0,1])+abs(K[0,1])), title='K12')
    plt.colorbar(p12)
    
    plt.subplot(223)
    p13 = plot(abs(K[1,0]-K_exa[1,0])*2*100/(abs(K_exa[1,0])+abs(K[1,0])), title='K21')
    plt.colorbar(p13)
    
    plt.subplot(224)
    p14 = plot(abs(K[1,1]-K_exa[1,1])*2*100/(abs(K_exa[1,1])+abs(K[1,1])), title='K22')
    plt.colorbar(p14)
    plt.suptitle('Percentage error of permeability tensor (%)')
    
    plt.savefig(path + 'K_error_per.eps')    
    
# plot convergence of K
def plot_K_conv(K11, K12, K21, K22, K_exa, path):
    plt.figure(figsize=(8.5,7))
    plt.plot(K11, label='$K^{(1,1)}_{mean}$')
    plt.plot(K12, label='$K^{(1,2)}_{mean}$')
    plt.plot(K21, label='$K^{(2,1)}_{mean}$')
    plt.plot(K22, label='$K^{(2,2)}_{mean}$')
    plt.axhline(y = K_exa[0],ls = "--", label = 'exact value')
    plt.axhline(y = K_exa[1],ls = "--")
    plt.axhline(y = K_exa[2],ls = "--")
    plt.axhline(y = K_exa[3],ls = "--")
    plt.legend(loc='lower left')
    plt.xlabel('Number of iterations in optimisation loop')
    plt.title('Convergence of K components (mean)')
    plt.savefig(path + 'K_conv.eps')
  
# plot objective function residual
def plot_fval(fun_val, path, filename='fval_conv.eps'):
    plt.figure(figsize=(8.5,7))
    plt.plot(fun_val, label='function value J')
    plt.xlabel('Number of iterations in optimisation loop')
    plt.title('Convergence of functional residual J')
    plt.savefig(path + filename)
    
# plot inf norm of gradient
def plot_grad(grad_val, path, filename='grad_conv.eps'):
    plt.figure(figsize=(8.5,7))
    plt.plot(grad_val, label='inf norm of projected gradient')
    plt.xlabel('Number of iterations in optimisation loop')
    plt.title('Convergence of inf norm of gradient')
    plt.savefig(path + filename)
    
############ RUN ############
if __name__ == '__main__':
    # unit mesh
    mesh = UnitSquareMesh(49, 49)
    # function spaces
    V = FunctionSpace(mesh, 'P', 1)
    W = TensorFunctionSpace(mesh, 'P', 1, shape=(2,2), symmetry=True)
    VL = VectorFunctionSpace(mesh, 'P', 1)
    # boundary condition of pressure
    u_D = Expression('1 - 2*x[0]*x[0] + 2*x[1]*x[1]', degree=2)  # exact sol
    bc1 = DirichletBC(V, u_D, 'on_boundary')
    # source term of velocity of fluid
    f = Constant(0)
    # exact permeability K (the 4th element '0' is dummy, when symm is true, only first 3 elements being read)
    K_exa = interpolate(Expression((('0.5*x[0]','0'), ('x[0]','0')), degree=1), W)  # permeability tensor     
    # time span
    T = 2.0
    # number of time steps
    num_steps = 2000
    # time step
    dt = T / num_steps   
    # diffusion coefficient
    D_coef = Constant(0.1)
    # boundary condition and source term of concentration
    q = Constant(0)
    c_D = Expression('x[0]*(1-x[0])*x[1]', degree=3)  # exact sol
    bc2 = DirichletBC(V, c_D, 'on_boundary')
    
    #################### generate synthetic data
    import os
    if os.path.exists('synData.h5'):
        if_continue = input("Synthetic data already exists, overwrite? [y/n]: ")
        if if_continue == "y":
            generate_synData(bc1, bc2, K_exa, f, V, VL, mesh, q, D_coef, dt, num_steps)
    else:
        generate_synData(bc1, bc2, K_exa, f, V, VL, mesh, q, D_coef, dt, num_steps)
            
    # prompt for continue
    if_continue = input("Synthetic data has been generated, start estimation? [y/n]: ")
    if if_continue == "y":
        # regularisation parameters
        alpha = 100
        beta = 1e-2   
        # time slicing factor
        slicing = 0.025
           
        # compute observed time step
        dt = dt / slicing
        # read synthetic concentration according to observed time step
        u_syn, synConc = read_synData(mesh, dt, num_steps, V, slicing)
        # number of observed timesteps
        num_steps = int(num_steps * slicing)
        # initial guess K0 (4th element is a dummy element since W is a symm tensorfunctionspace)
        K = interpolate(Constant([[1, 0], [1, 0]]), W)
        
        #################### start estimation
        while beta > 5e-7:
            # save path
            path = 'beta%.1E/' %beta
            # create folder        
            if not os.path.exists(path[:-1]):
                os.mkdir(path[:-1])
            
            if os.path.exists(path + 'K.h5'):
                if_continue = input("Estimated permeability (param beta=%.1E) has been found, overwrite? [y/n]: " %beta)
                if if_continue == "y":                
                    # run estimation
                    K = estimate_K(mesh, V, VL, K, K_exa, bc1, bc2, f, q, D_coef, dt, num_steps, u_syn, synConc, alpha, beta, path)
                else:
                    # load previously estimated permeability
                    hdf = HDF5File(mesh.mpi_comm(), path + 'K.h5', "r")
                    hdf.read(K, "K")
                    hdf.close()
            else:
                # run estimation
                K = estimate_K(mesh, V, VL, K, K_exa, bc1, bc2, f, q, D_coef, dt, num_steps, u_syn, synConc, alpha, beta, path)

            beta *= 1e-3
          