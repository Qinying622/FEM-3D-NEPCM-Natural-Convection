# Step 1A: Natural convection flow of an air-filled cavity containing two
# circular heated cylinders
# 
# Objective:
#
# Establish a verified baseline for natural convection around two internal
# heated cylinders
#
# The implementation uses Taylor-Hood Q2-Q1 element for momentum and continuity
# and Q2 for temperature.
#
# This code is run using DOLFINx v0.11.0
#
# Copyright (C) 2026 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, geometry
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_rectangle, locate_entities_boundary, meshtags
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    CellDiameter,
    FacetNormal,
    Measure,
    TestFunction,
    TrialFunction,
    TestFunctions,
    TrialFunctions,
    derivative,
    div,
    dot,
    dx,
    grad,
    inner,
    split,
    sqrt,
)
