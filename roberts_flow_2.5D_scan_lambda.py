"""
Dedalus script for kinematic dynamo, drawing from:

Roberts, G.O., 1972,
``Dynamo action of fluid motions with two-dimensional periodicity''

Usage:
    roberts_flow1.py [options]

Options:
    --N=<N>           Resolution in y, z [default: 16]

    --min_lambda=<λm>    Min resistivity [default: 1/64]
    --max_lambda=<λm>    Max resistivity [default: 1]

    --flow=<flow>     Roberts flow to study [default: 1]

    --min_kx=<minkx>  Min kx to study [default: 0.1]
    --max_kx=<maxkx>  Max kx to study [default: 1]
    --n_kx=<n_kx>     How many kxs to sample (int) [default: 20]

    --aspect=<a>      Horizontal number of patterns in domain (integer) [default: 1]

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

aspect = int(args['--aspect'])

N = int(args['--N'])
Nx = Ny = Nz = N*aspect

N_evals = int(float(args['--eigs']))
target = float(args['--target'])

min_kx = float(args['--min_kx'])
max_kx = float(args['--max_kx'])
n_kx = int(float(args['--n_kx']))

λ_min = float(Fraction(args['--min_lambda']))
λ_max = float(Fraction(args['--max_lambda']))

log_λ_min = int(np.log2(λ_min))
log_λ_max = int(np.log2(λ_max))

n_λ = log_λ_max - log_λ_min + 1
λs = np.logspace(log_λ_min, log_λ_max, base=2, num=n_λ)
logger.info(λs)

flow = int(args['--flow'])

Ly = Lz = 2*np.pi*aspect

data_dir = 'roberts_flow{:}_N{:d}_lambda_scan'.format(flow, N)
if args['--dense']:
    data_dir += '_dense'

if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))

dealias = 3/2
dtype = np.complex128

coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
ybasis = de.ComplexFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = de.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

bases = (ybasis, zbasis)

# Fields
ω = dist.Field(name='ω')
φ = dist.Field(name='φ', bases=bases)
A = dist.VectorField(coords, name='A', bases=bases)
τ_φ = dist.Field(name='τ_φ')

#x = xbasis.local_grid(1)
y = ybasis.local_grid(1)
z = zbasis.local_grid(1)

kx = dist.Field(name='kx')

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

ex, ey, ez = coords.unit_vector_fields(dist)

# follows Roberts 1972 convention, eq 1.1, 2.8
dt = lambda A: ω*A
dx = lambda A: 1j*kx*A
dy = lambda A: de.Differentiate(A, coords['y'])
dz = lambda A: de.Differentiate(A, coords['z'])

#grad = lambda A: de.Gradient(A, coords) #+ 1j*kx*A*ex
div = lambda A:  dx(A@ex) + dy(A@ey) + dz(A@ez)
grad = lambda A: dx(A)*ex + dy(A)*ey + dz(A)*ez
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))
curl = lambda A: (dy(A@ez)-dz(A@ey))*ex + (dz(A@ex)-dx(A@ez))*ey + (dx(A@ey)-dy(A@ex))*ez

problem = de.EVP([A, φ, τ_φ], eigenvalue=ω, namespace=locals())
problem.add_equation("dt(A) + grad(φ) - λ*lap(A) - cross(u, curl(A)) = 0")
problem.add_equation("div(A) + τ_φ = 0")
problem.add_equation("integ(φ) = 0")
solver = problem.build_solver()

dlog = logging.getLogger('subsystems')
dlog.setLevel(logging.WARNING)

def peak_growth(λ_i, kx_i):
    λ['g'] = λ_i
    kx['g'] = kx_i
    if args['--dense']:
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
    else:
        solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]
    logger.info('kx = {:}, ω = {:}'.format(kx['g'], evals[-1]))
    # turn max into min, for minimize
    return evals[-1]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

for λ_i in λs:
    logger.info('solving for λ={:}'.format(λ_i))
    eigs = []
    kxs = np.geomspace(min_kx, max_kx, num=n_kx)
    for kx_i in kxs:
        eigs.append(peak_growth(λ_i, kx_i))
    eigs = np.array(eigs)
    kxs = np.array(kxs)
    mask = eigs.real > 0
    ax.plot(kxs[mask], eigs.real[mask], label=r'$\lambda={:}$'.format(λ_i))
ax.axhline(y=0, alpha=0.5, color='xkcd:grey')
ax.set_xlabel('kx')
ax.set_xscale('log')
ax.set_ylabel(r'$\omega_R$')
ax.legend()
fig.savefig(data_dir+'/evals_scan_omega_kx.png', dpi=300)
