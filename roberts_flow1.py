"""
Dedalus script for kinematic dynamo, drawing from:

Roberts, G.O., 1972,
``Dynamo action of fluid motions with two-dimensional periodicity''

Usage:
    roberts_flow1.py [options]

Options:
    --N=<N>           Resolution in x, y, z [default: 16]
    --lambda=<λ>      Resistivity [default: 1/8]
    --dense           Solve densely for all eigenvalues (slow)

    --verbose         Show plots on screen
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
import dedalus.public as de
from fractions import Fraction

from docopt import docopt
args = docopt(__doc__)

N = int(args['--N'])
Nx = Ny = Nz = N

λ_in = float(Fraction(args['--lambda']))

dealias = 3/2
dtype = np.complex128

Lx = Ly = Lz = 2*np.pi

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

u['g'][0] = np.cos(y) - np.cos(z)
u['g'][1] = np.sin(z)
u['g'][2] = np.sin(y)

λ = dist.Field(name='λ')
λ['g'] = λ_in

dt = lambda A: 1j*ω*A
grad = lambda A: de.Gradient(A, coords)

problem = de.EVP([A, φ, τ_φ], eigenvalue=ω, namespace=locals())
problem.add_equation("dt(A) + grad(φ) - λ*lap(A) - cross(u, curl(A)) = 0")
problem.add_equation("div(A) + τ_φ = 0")
problem.add_equation("integ(φ) = 0")
solver = problem.build_solver()

if args['--dense']:
    solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
else:
    solver.solve_sparse(solver.subproblems[0], N=10, target=0.5, rebuild_matrices=True)
i_evals = np.argsort(solver.eigenvalues.real)
if args['--dense']:
    i_evals = i_evals[0:len(i_evals)//2]
# else:
#     i_evals = i_evals[-7:]
evals = solver.eigenvalues[i_evals]
ks = np.arange(10)+1
print(evals)

import matplotlib.pyplot as plt

n_modes = 4 #8
i_modes = slice(-n_modes*2,None,2)
fig, ax = plt.subplots(nrows=(n_modes+1), ncols=2, figsize=[12,6])
ax[0,0].plot(ks, evals.real, linestyle='none', marker='o')
ax[0,1].plot(ks, evals.imag, linestyle='none', marker='o')

# reduce number of plotted eigenmodes
i_evals = i_evals[i_modes]
evals = solver.eigenvalues[i_evals]
ax[0,0].plot(ks[i_modes], evals.real, linestyle='none', marker='o')
ax[0,1].plot(ks[i_modes], evals.imag, linestyle='none', marker='o')

ax[0,0].set_ylabel(r'$\omega_R$')
ax[0,1].set_ylabel(r'$\omega_I$')
curl = lambda A: de.Curl(A)
B_op = curl(A)
for n, idx in enumerate(i_evals, start=1):
    solver.set_state(idx, solver.subsystems[0])
    # B = B_op.evaluate()
    # B.change_scales(1)
    i_max = np.unravel_index(np.argmax(np.abs(A['g'][0]), axis=None), A['g'][0].shape)
    norm = A['g'][0][i_max]
    print(i_max, norm)
    Ag = (A['g']/norm).real
    #ax[1].plot(x[:,0,0], Bg[0][:,0,0]/np.max(np.abs(Bg)), label=f"n={n}")
    print(np.min(Ag[0].imag), np.max(Ag[0].imag))
    #ax[n].pcolormesh(y[0,:,0], z[0,0,:], Ag[0][0,:,:])
    ax[n,0].pcolormesh(x[:,0,0], z[0,0,:], Ag[0][:,0,:], cmap='RdYlBu_r')
    ax[n,1].pcolormesh(y[0,:,0], z[0,0,:], Ag[0][0,:,:], cmap='RdYlBu_r')
    ax[n,0].set_ylabel('$\omega_R$={:.2f}'.format(evals[n-1].real))
    ax[n,1].set_ylabel('$\omega_I$={:.2f}'.format(evals[n-1].imag))

#fig.tight_layout()
if args['--verbose']:
    plt.show()
fig.savefig('evals_roberts_flow1.png', dpi=300)
