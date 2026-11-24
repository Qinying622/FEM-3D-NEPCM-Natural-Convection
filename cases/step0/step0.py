# Step 0: Natural convection flow in a square enclosure
# Objective:
#
# Establish that the coupled flow-temperature solver reproduces standard
# natural convection behavior in a square cavity.
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

# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, ((0, 0), (1, 1)), (128, 128), CellType.quadrilateral,
    ghost_mode=GhostMode.none
)

# Create function spaces
P2 = element(
    "Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=default_real_type,
)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
Q2 = element("Lagrange", msh.basix_cell(), 2, dtype=default_real_type)

# Create mixed element for velocity, pressure and temperature
ME = mixed_element([P2, P1, Q2])
W = functionspace(msh, ME)

# Get each boundary facets
facets0 = locate_entities_boundary(
    msh, msh.geometry.dim - 1, lambda x: np.isclose(x[1], 0)
)
facets1 = locate_entities_boundary(
    msh, msh.geometry.dim - 1, lambda x: np.isclose(x[0], 1)
)
facets2 = locate_entities_boundary(
    msh, msh.geometry.dim - 1, lambda x: np.isclose(x[1], 1)
)
facets3 = locate_entities_boundary(
    msh, msh.geometry.dim - 1, lambda x: np.isclose(x[0], 0)
)

facets = np.hstack((facets0, facets1, facets2, facets3))

# Apply Dirichlet BC
# Velocity
W0 = W.sub(0)
V, _ = W0.collapse()
noslip = Function(V)
noslip.x.petsc_vec.set(0.0)
bc_noslip = dirichletbc(
    noslip, locate_dofs_topological((W0, V), msh.geometry.dim-1, facets), W0
)

# Pressure (prevent singular system)
W1 = W.sub(1)
Q, _ = W1.collapse()
pressure = Function(Q)
pressure_point = locate_dofs_geometrical(
    Q,
    lambda x: np.logical_and(np.isclose(x[0], 0.), np.isclose(x[1], 0.))
)
bc_p = dirichletbc(
    pressure,
    locate_dofs_topological((W1, Q), msh.geometry.dim-1, pressure_point),
    W1)

# Temperature
W2 = W.sub(2)
P, _ = W2.collapse()
T_left = Function(P)
T_left.x.array[:] = 1.0
bc_Tl = dirichletbc(
    T_left,
    locate_dofs_topological((W2, P), msh.geometry.dim-1, facets3),
    W2
)
T_right = Function(P)
T_right.x.array[:] = 0.0
bc_Tr = dirichletbc(
    T_right,
    locate_dofs_topological((W2, P), msh.geometry.dim-1, facets1),
    W2
)

bcs = [bc_noslip, bc_p, bc_Tl, bc_Tr]

# Define variational problem
(u, p, T) = TrialFunctions(W)
wk = Function(W)
uk, pk, Tk = split(wk)
(v, q, theta) = TestFunctions(W)
Pr = Constant(msh, PETSc.ScalarType(0.71))
Ra = Constant(msh, PETSc.ScalarType(1e3))
ey = Constant(msh, (PETSc.ScalarType(0.0), PETSc.ScalarType(1.0)))

# SUPG stabilisation
h = CellDiameter(msh)
u_norm = sqrt(dot(uk, uk) + 1e-12)
tau = 1 / (2*u_norm/h + 4/(h**2))

F = (
    Pr * inner(grad(uk), grad(v)) * dx
    + inner(dot(grad(uk), uk), v) * dx
    - inner(pk, div(v)) * dx
    + inner(q, div(uk)) * dx
    - Pr*Ra * inner(Tk, dot(ey, v)) * dx
    + inner(dot(uk, grad(Tk)), theta) * dx
    + inner(grad(Tk), grad(theta)) * dx
    + tau * dot(uk, grad(theta)) * dot(uk, grad(Tk)) * dx
    + tau * inner(grad(dot(uk, grad(theta))), grad(Tk)) * dx
)

J = derivative(F, wk)

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
    "snes_atol": 1e-15,
    "snes_rtol": 1e-15,
    "snes_max_it": 25,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None,
}

fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
fig_Nu, ax_Nu = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
Rayleighs = [1e3, 1e4, 1e5, 1e6]
for Rayleigh in Rayleighs:
    print(f"Solving for Ra = {Rayleigh}")
    Ra.value = Rayleigh

    # Instantiate problem and solver
    problem = NonlinearProblem(F, wk, bcs=bcs, J=J, 
                            petsc_options_prefix="step0_", 
                            petsc_options=petsc_options)
    # Solve
    log.set_log_level(log.LogLevel.WARNING)
    _ = problem.solve()
    converged_reason = problem.solver.getConvergedReason()
    assert converged_reason > 0
    num_iterations = problem.solver.getIterationNumber()
    print(f"Number of interations: {num_iterations:d}")

    uh, ph, Th = wk.sub(0).collapse(), wk.sub(1).collapse(), wk.sub(2).collapse()

    with XDMFFile(msh.comm, f"result/velocity_{Rayleigh}.xdmf", "w") as ufile_xdmf:
        uh.x.scatter_forward()
        P1 = element(
            "Lagrange",
            msh.basix_cell(),
            1,
            shape=(msh.geometry.dim,),
            dtype=default_real_type,
        )
        u1 = Function(functionspace(msh, P1))
        u1.interpolate(uh)
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u1)

    with XDMFFile(msh.comm, f"result/pressure_{Rayleigh}.xdmf", "w") as pfile_xdmf:
        ph.x.scatter_forward()
        Q1 = element(
            "Lagrange",
            msh.basix_cell(),
            1,
            dtype=default_real_type,
        )
        p1 = Function(functionspace(msh, Q1))
        p1.interpolate(ph)
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p1)

    with XDMFFile(msh.comm, f"result/temperature_{Rayleigh}.xdmf", "w") as tfile_xdmf:
        Th.x.scatter_forward()
        Q1 = element(
            "Lagrange",
            msh.basix_cell(),
            1,
            dtype=default_real_type,
        )
        T1 = Function(functionspace(msh, Q1))
        T1.interpolate(Th)
        tfile_xdmf.write_mesh(msh)
        tfile_xdmf.write_function(T1)

    # Compute average Nusselt
    n = FacetNormal(msh)
    facet_tags = meshtags(
        msh, msh.topology.dim-1, facets3, np.ones_like(facets3, dtype=np.int32))
    ds_left = Measure("ds", domain=msh, subdomain_data=facet_tags)
    arc_length = msh.comm.allreduce(
        assemble_scalar(form(Constant(msh, PETSc.ScalarType(1.0)) * ds_left(1))),
        op=MPI.SUM)
    Nu = msh.comm.allreduce(
        assemble_scalar(form(dot(grad(Th), n) * ds_left(1))), op=MPI.SUM)
    Nu_avg = Nu / arc_length
    print(f"Average Nusselt number on hot boundary: {Nu_avg}")

    # Get the maximum horizontal velocity at x = 0.5
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)

    tol = 0.001  #
    n_sample = 201
    y_sample = np.linspace(0. + tol, 1 - tol, n_sample)
    points = np.zeros((3, n_sample))
    points[0] = 0.5
    points[1] = y_sample

    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = uh.eval(points_on_proc, cells)
    p_values = pressure.eval(points_on_proc, cells)

    print(f"u_max at x = 0.5: {u_values[:, 0].max()}")

    ax[0].plot(y_sample, u_values[:, 0], label=f"Ra={Rayleigh:.0e}")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel("u")
    ax[0].set_title("Horizontal velocity at x = 0.5")
    ax[0].legend()

    # Get the maximum vertical velocity at y = 0.5
    x_sample = np.linspace(0. + tol, 1 - tol, n_sample)
    points = np.zeros((3, n_sample))
    points[0] = x_sample
    points[1] = 0.5

    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = uh.eval(points_on_proc, cells)
    p_values = pressure.eval(points_on_proc, cells)

    print(f"v_max at y = 0.5: {u_values[:, 1].max()}")

    ax[1].plot(x_sample, u_values[:, 1], label=f"Ra={Rayleigh:.0e}")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("v")
    ax[1].set_title("Vertical velocity at y = 0.5")
    ax[1].legend()

    # Compute heat balance
    n = FacetNormal(msh)
    facet_tags_h = meshtags(
        msh, msh.topology.dim-1, facets0, np.ones_like(facets0, dtype=np.int32))
    ds_h = Measure("ds", domain=msh, subdomain_data=facet_tags_h)
    arc_length_h = msh.comm.allreduce(
        assemble_scalar(form(Constant(msh, PETSc.ScalarType(1.0)) * ds_h(1))),
        op=MPI.SUM)
    Q_h = msh.comm.allreduce(
        assemble_scalar(form(dot(grad(Th), n) * ds_h(1))), op=MPI.SUM)
    Nu_h = Q_h / arc_length_h
    print(f"Heat entering through the hot wall: {Q_h}")
    print(f"Average Nusselt number through the hot wall: {Nu_h}")

    facet_tags_c = meshtags(
        msh, msh.topology.dim-1, facets2, np.ones_like(facets2, dtype=np.int32))
    ds_c = Measure("ds", domain=msh, subdomain_data=facet_tags_c)
    arc_length_c = msh.comm.allreduce(
        assemble_scalar(form(Constant(msh, PETSc.ScalarType(1.0)) * ds_c(1))),
        op=MPI.SUM)
    Q_c = msh.comm.allreduce(
        - assemble_scalar(form(dot(grad(Th), n) * ds_c(1))), op=MPI.SUM)
    Nu_c = Q_c / arc_length_c
    print(f"Heat exiting through cold wall: {Q_c}")
    print(f"Average Nusselt number through the cold wall: {Nu_c}")

    # Plot Nusselt number along the hot wall
    V_flux = functionspace(msh, ("DG", 2))
    Nu_hot = Function(V_flux)

    expr = Expression(-Th.dx(0), V_flux.element.interpolation_points)
    Nu_hot.interpolate(expr)
    Nu_hot.x.scatter_forward()

    y_hot = np.linspace(1e-5, 1.0 - 1e-10, 501)

    points = np.column_stack((
        np.full_like(y_hot, 1e-5), 
        y_hot, 
        np.zeros_like(y_hot))).astype(np.float64)

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    hits = geometry.compute_colliding_cells(msh, candidates, points)

    if any(len(hits.links(i)) == 0 for i in range(len(points))):
        raise RuntimeError("Some sampling points could not be located.")

    cells = np.array(
        [hits.links(i)[0] for i in range(len(points))],
        dtype=np.int32,
    )

    Nu_values = Nu_hot.eval(points, cells).real.ravel()
    ax_Nu.plot(y_hot, Nu_values, label=f"Ra={Rayleigh:.0e}")
    ax_Nu.set(xlabel="$Y$", ylabel=r"$Nu_h(Y)$", xlim=(0, 1))
    ax_Nu.legend(frameon=False)
    ax_Nu.grid(alpha=0.25)

fig.savefig(f"result/step0_velocity_profiles.pdf")
fig.savefig(f"result/step0_velocity_profiles.png", dpi=300)
fig_Nu.tight_layout()
fig_Nu.savefig("result/local_nusselt_hot_wall.pdf")
fig_Nu.savefig("result/local_nusselt_hot_wall.png", dpi=300)

# ------------------------------ #
# Plot streamlines and isotherms #
# ------------------------------ #
n_sample = 201
x_sample = np.linspace(0., 1., n_sample)
y_sample = np.linspace(0., 1., n_sample)
X, Y = np.meshgrid(x_sample, y_sample)

points = np.column_stack((
    X.ravel(), Y.ravel(),
    np.zeros(X.size)
)).astype(np.float64)

# Find a containing mesh cell for each plotting point.
tree = geometry.bb_tree(msh, msh.topology.dim)
candidates = geometry.compute_collisions_points(tree, points)
collisions = geometry.compute_colliding_cells(
    msh, candidates, points
)

cells = np.full(len(points), -1, dtype=np.int32)

for i in range(len(points)):
    containing_cells = collisions.links(i)
    if len(containing_cells) > 0:
        cells[i] = containing_cells[0]

if np.any(cells < 0):
    raise RuntimeError("Some plotting points lie outside the local mesh.")

# Streamfunction space and homogeneous boundary condition.
V_psi = functionspace(msh, ("Lagrange", 2))

boundary_facets = locate_entities_boundary(
    msh,
    msh.topology.dim - 1,
    lambda x: (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    ),
)

boundary_dofs = locate_dofs_topological(
    V_psi, msh.topology.dim - 1, boundary_facets
)

bc_psi = dirichletbc(
    PETSc.ScalarType(0.0), boundary_dofs, V_psi
)

psi_trial = TrialFunction(V_psi)
psi_test = TestFunction(V_psi)

vorticity = uh[1].dx(0) - uh[0].dx(1)

a_psi = inner(
    grad(psi_trial), grad(psi_test)
) * dx

L_psi = vorticity * psi_test * dx

psi_problem = LinearProblem(
    a_psi,
    L_psi,
    bcs=[bc_psi],
    petsc_options_prefix="streamfunction_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)

psi_h = psi_problem.solve()
psi_h.x.scatter_forward()

psi_plot = psi_h.eval(points, cells).reshape(X.shape)
theta_plot = Th.eval(points, cells).reshape(X.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False, 
                        constrained_layout=True)

temperature_levels = np.linspace(0, 1.0, 11)

psi_min = float(psi_plot.min())
psi_max = float(psi_plot.max())
print(f"Streamfunction min: {psi_min}, max: {psi_max}")

if psi_max > psi_min:
    psi_levels = np.linspace(psi_min, psi_max, 17)
    axes[0, 0].contourf(X, Y, psi_plot, levels=psi_levels, cmap="coolwarm")
    axes[0, 0].contour(X, Y, psi_plot, levels=psi_levels[1:-1], colors="black",
                    linewidths=0.5, linestyles="solid")

psi_abs_max = np.max(np.abs(psi_plot))
axes[0, 0].set_title(
    rf"$Ra={Rayleigh:.0e}$: streamlines"
    "\n"
    rf"$\max|\psi|\approx {psi_abs_max:.3f}$"
)
axes[0, 0].set_aspect("equal")
axes[0, 0].set_xlim(0.0, 1.0)
axes[0, 0].set_ylim(0.0, 1.0)
axes[0, 0].set_xlabel("$X$")
axes[0, 0].set_ylabel("$Y$")

filled = axes[0, 1].contourf(
    X, Y, theta_plot,
    levels=temperature_levels,
    cmap="coolwarm",
    vmin=0.0,
    vmax=1.0,
)
axes[0, 1].contour(
    X, Y, theta_plot,
    levels=temperature_levels[1:-1],
    colors="black",
    linewidths=0.5,
)
axes[0, 1].set_title(rf"$Ra={Rayleigh:.0e}$: isotherms")
axes[0, 1].set_aspect("equal")
axes[0, 1].set_xlim(0.0, 1.0)
axes[0, 1].set_ylim(0.0, 1.0)
axes[0, 1].set_xlabel("$X$")
axes[0, 1].set_ylabel("$Y$")

fig.colorbar(
    filled,
    ax=axes[:, 1].tolist(),
    label=r"$\theta$",
)

plt.savefig(f"result/step0_streamfunction_isotherms_{Rayleigh:.0e}.pdf")
plt.savefig(f"result/step0_streamfunction_isotherms_{Rayleigh:.0e}.png", dpi=300)
plt.close(fig)
