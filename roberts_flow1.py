"""
Dedalus script for kinematic dynamo, drawing from:

Roberts, G.O., 1972,
``Dynamo action of fluid motions with two-dimensional periodicity''

Usage:
    roberts_flow1.py [options]

Options:
    --N=<N>           Resolution in x, y, z [default: 16]
    --lambda=<λ>      Resistivity [default: 1/8]
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

#solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
solver.solve_sparse(solver.subproblems[0], N=20, target=0.25, rebuild_matrices=True)
i_evals = np.argsort(solver.eigenvalues.real)
evals = solver.eigenvalues[i_evals]
print(evals)
