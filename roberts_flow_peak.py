"""
Dedalus script for kinematic dynamo, drawing from:

Roberts, G.O., 1972,
``Dynamo action of fluid motions with two-dimensional periodicity''

This script solves in 2.5D and optimizes to find the peak growth rate.

Usage:
    roberts_flow1.py [options]

Options:
    --N=<N>           Resolution in y, z [default: 16]
    --lambda=<λ>      Resistivity [default: 1/8]
    --flow=<flow>     Roberts flow to study [default: 1]

    --kx=<kx>         Target kx to study [default: 0.55]

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

λ_in = float(Fraction(args['--lambda']))

flow = int(args['--flow'])
kx_start = float(args['--kx'])

Ly = Lz = 2*np.pi*aspect

data_dir = 'roberts_flow{:}_N{:d}_lambda{:}'.format(flow, N, λ_in)
if args['--dense']:
    data_dir += '_dense'

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

u = dist.VectorField(coords, name='u', bases=bases)

kx = dist.Field(name='kx')

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

def peak_growth(kx_i):
    kx['g'] = kx_i
    if args['--dense']:
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
    else:
        solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]
    logger.info('kx = {:}, ω = {:}'.format(kx['g'], evals[-1]))
    # turn max into min, for minimize
    return -1*evals[-1].real

import scipy.optimize as sciop

logger.info('starting optimization solve')
result = sciop.minimize(peak_growth, kx_start)
logger.info('solve complete')
if result.success:
    logger.info('fastest growing mode, ω = {:} at kx = {:}'.format(-1*result.fun, result.x[0]))
else:
    logger.info('solver failed to converge: {}'.format(result.message))
