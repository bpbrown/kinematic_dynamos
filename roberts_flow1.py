"""
Dedalus script for kinematic dynamo, drawing from:

Roberts, G.O., 1972,
``Dynamo action of fluid motions with two-dimensional periodicity''

Usage:
    roberts_flow1.py [options]

Options:
    --N=<N>           Resolution in x, y, z [default: 16]
    --lambda=<λ>      Resistivity [default: 1/8]
    --flow=<flow>     Roberts flow to study [default: 1]

    --aspect=<a>      Aspect ratio, Lx/Lz, with Ly=Lz [default: 1]

    --target=<targ>   Target value for sparse eigenvalue search [default: 0.2]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --dense           Solve densely for all eigenvalues (slow)

    --verbose         Show plots on screen
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
import dedalus.public as de
from fractions import Fraction

import os

from docopt import docopt
args = docopt(__doc__)

N = int(args['--N'])
Nx = Ny = Nz = N

N_evals = int(float(args['--eigs']))
target = float(args['--target'])

λ_in = float(Fraction(args['--lambda']))

flow = int(args['--flow'])
aspect = float(args['--aspect'])

Lx = Ly = Lz = 2*np.pi

data_dir = 'roberts_flow{:}_N{:d}_lambda{:}'.format(flow, N, λ_in)
if args['--dense']:
    data_dir += '_dense'
if aspect != 1:
    data_dir += '_a{:s}'.format(args['--aspect'])
    Lx *= aspect
    Nx = int(Nx*aspect)

if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))

dealias = 3/2
dtype = np.complex128



coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
xbasis = de.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = de.ComplexFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = de.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

bases = (xbasis, ybasis, zbasis)

# Fields
ω = dist.Field(name='ω')
φ = dist.Field(name='φ', bases=bases)
A = dist.VectorField(coords, name='A', bases=bases)
τ_φ = dist.Field(name='τ_φ')

x = xbasis.local_grid(1)
y = ybasis.local_grid(1)
z = zbasis.local_grid(1)

u = dist.VectorField(coords, name='u', bases=bases)

if flow == 1:
    u['g'][0] = np.cos(y) - np.cos(z)
    u['g'][1] = np.sin(z)
    u['g'][2] = np.sin(y)
elif flow == 2:
    u['g'][0] = np.cos(y) + np.cos(z)
    u['g'][1] = np.sin(z)
    u['g'][2] = np.sin(y)
elif flow == 3:
    u['g'][0] = 2*np.cos(y)*np.cos(z)
    u['g'][1] = np.sin(z)
    u['g'][2] = np.sin(y)
elif flow == 4:
    u['g'][0] = np.sin(y+z)
    u['g'][1] = np.sin(2*z)
    u['g'][2] = np.sin(2*y)
else:
    raise ValueError('flow = {} not a valid choice (flow = {1,2,3,4})'.format(flow))


λ = dist.Field(name='λ')
λ['g'] = λ_in

grad = lambda A: de.Gradient(A, coords)

# follows Roberts 1972 convention, eq 2.8
dt = lambda A: ω*A

problem = de.EVP([A, φ, τ_φ], eigenvalue=ω, namespace=locals())
problem.add_equation("dt(A) + grad(φ) - λ*lap(A) - cross(u, curl(A)) = 0")
problem.add_equation("div(A) + τ_φ = 0")
problem.add_equation("integ(φ) = 0")
solver = problem.build_solver()

if args['--dense']:
    solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
else:
    solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
i_evals = np.argsort(solver.eigenvalues.real)
if args['--dense']:
    i_evals = i_evals[0:len(i_evals)//2]
# else:
#     i_evals = i_evals[-7:]
evals = solver.eigenvalues[i_evals]
ks = np.arange(N_evals)+1
print(evals)

import matplotlib.pyplot as plt


n_modes = N_evals #4
step = 1
# reduce number of plotted eigenmodes
i_modes = slice(None,-(n_modes*step+1),-step)

kxs = []
eigs = []
for i, idx in enumerate(i_evals[i_modes]):
    gs_kw = dict(height_ratios=[1, 6])
    fig, axd = plt.subplot_mosaic([['eval_R', 'eval_I'],
                                   ['xz', 'yz']],
                                   gridspec_kw=gs_kw, figsize=[12,6],
                                   layout="constrained")

    axd['eval_R'].scatter(ks, evals.real)
    axd['eval_I'].scatter(ks, evals.imag)

    eval = solver.eigenvalues[idx]
    axd['eval_R'].scatter(ks[i_modes][i], eval.real)
    axd['eval_I'].scatter(ks[i_modes][i], eval.imag)
    fig.suptitle('$\omega_R$={:.2f}'.format(eval.real)+' $\omega_I$={:.2f}'.format(eval.imag))

    axd['eval_R'].set_ylabel(r'$\omega_R$')
    axd['eval_I'].set_ylabel(r'$\omega_I$')

    axd['eval_R'].axhline(y=0, alpha=0.5, color='xkcd:grey')

    curl = lambda A: de.Curl(A)
    B_op = curl(A)
    solver.set_state(idx, solver.subsystems[0])
    # B = B_op.evaluate()
    # B.change_scales(1)
    i_max = np.unravel_index(np.argmax(np.abs(A['g'][0]), axis=None), A['g'][0].shape)
    norm = A['g'][0][i_max]
    Ag = (A['g']/norm).real
    #ax[1].plot(x[:,0,0], Bg[0][:,0,0]/np.max(np.abs(Bg)), label=f"n={n}")
    #ax[n].pcolormesh(y[0,:,0], z[0,0,:], Ag[0][0,:,:])
    axd['xz'].pcolormesh(x[:,0,0], z[0,0,:], Ag[0][:,0,:].T, cmap='RdYlBu_r')
    axd['yz'].pcolormesh(y[0,:,0], z[0,0,:], Ag[0][0,:,:].T, cmap='RdYlBu_r')
    axd['xz'].set_xlabel('x')
    axd['xz'].set_ylabel('z')
    axd['yz'].set_xlabel('y')
    axd['yz'].set_ylabel('z')
    axd['xz'].set_aspect(1)
    axd['yz'].set_aspect(1)
    fig.savefig(data_dir+'/evals_mode{:03d}.png'.format(i), dpi=300)
    plt.close(fig)

    kx = dist.coeff_layout.local_group_arrays(xbasis.domain, scales=1)
    ky = dist.coeff_layout.local_group_arrays(ybasis.domain, scales=1)
    kz = dist.coeff_layout.local_group_arrays(zbasis.domain, scales=1)

    fig, axd = plt.subplot_mosaic([['kx kz 0', 'ky kz 0']],
                                   figsize=[12,6],
                                   layout="constrained")
    # axd['kx kz 0'].pcolormesh(kx[0][:,0,0], kz[2][0,0,:], np.abs(A['c'][0][:,0,:]), shading='nearest')
    # axd['ky kz 0'].pcolormesh(ky[1][0,:,0], kz[2][0,0,:], np.abs(A['c'][0][0,:,:]), shading='nearest')
    axd['kx kz 0'].imshow(np.abs(A['c'][0][:,0,:]).T)
    axd['ky kz 0'].imshow(np.abs(A['c'][0][0,:,:]).T)
    axd['kx kz 0'].set_xlabel('kx')
    axd['kx kz 0'].set_ylabel('kz')
    axd['ky kz 0'].set_xlabel('ky')
    axd['ky kz 0'].set_ylabel('kz')
    axd['kx kz 0'].set_aspect(1)
    axd['ky kz 0'].set_aspect(1)
    #i_max = np.unravel_index(np.argmax(np.abs(A['c'][0]), axis=None), A['c'][0].shape)
    i_max = np.argmax(np.abs(A['c'][0][:,0,:]))
    ij_max = np.unravel_index(i_max, A['c'][0][:,0,:].shape)
    kx_max = kx[0][ij_max[0],0,0]#*2*np.pi/Lx
    kz_max = kz[2][0,0,ij_max[1]]#*2*np.pi/Lz
    print('{:.4g}, {:.2g}i @ kx={}, kz={}'.format(eval.real, eval.imag, kx_max, kz_max))
    fig.savefig(data_dir+'/evals_coeffs_mode{:03d}.png'.format(i), dpi=300)
    plt.close(fig)

    kxs.append(kx_max)
    eigs.append(eval)

eigs = np.array(eigs)
kxs = np.array(kxs)
fig, ax = plt.subplots()
ax.scatter(kxs, eigs.real, label=r'$\omega_R$')
ax.scatter(kxs, eigs.imag, label=r'$\omega_I$')
ax.axhline(y=0, alpha=0.5, color='xkcd:grey')
ax.set_xlabel('kx')
ax.set_ylabel('omega')
ax.legend()
fig.savefig(data_dir+'/evals_omega_kx.png', dpi=300)
