"""
Tube MPC Demo — Robust Model Predictive Control with mpt4py
============================================================

Companion script for Chapter 10 (Robust MPC & Tube MPC) of the
MPC lecture notes by I. Kucukdemiral.

This script demonstrates the complete Tube MPC design pipeline:
  1. LQR feedback gain computation
  2. Minimal robust positively invariant (mRPI) set computation
  3. Constraint tightening (Pontryagin difference)
  4. Terminal set computation (maximal invariant set)
  5. Tube MPC formulation and closed-loop simulation with disturbances

System: Double integrator  x+ = Ax + Bu + w
    x = [position, velocity],  u = acceleration,  w ∈ W (bounded)

Requirements:
    pip install numpy matplotlib control cvxpy piqp \
        --find-links https://github.com/PREDICT-EPFL/MPC-Course-EPFL/releases/expanded_assets/mpt4py_wheels \
        --find-links https://github.com/PREDICT-EPFL/MPC-Course-EPFL/releases/expanded_assets/pycddlib_wheels \
        mpt4py==0.1.5

Usage:
    python tube_mpc_demo.py

Code:  https://kucukdemiral.github.io/tube_mpc_demo.py
Colab: https://kucukdemiral.github.io/tube_mpc_demo.ipynb

I. Kucukdemiral — Glasgow Caledonian University
"""

import numpy as np
import matplotlib.pyplot as plt
from mpt4py import Polyhedron
from control import dlqr
import cvxpy as cp

# ══════════════════════════════════════════════════════════════════
#  USER SETTINGS — modify these to experiment
# ══════════════════════════════════════════════════════════════════
x_max = 3.0       # state constraint bound |x_i| <= x_max
u_max = 0.5       # input constraint bound |u| <= u_max
w_max = 0.1       # disturbance bound |w_i| <= w_max

N_horizon = 10    # MPC prediction horizon
N_sim = 40        # closed-loop simulation steps
N_samples = 20    # number of Monte Carlo disturbance realisations

Q_weight = np.eye(2)         # state cost weight
R_weight = 10 * np.eye(1)    # input cost weight (large = less aggressive)

# ══════════════════════════════════════════════════════════════════
#  STEP 0: System definition
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Tube MPC Demo — Double Integrator")
print("=" * 60)

A = np.array([[1, 1],
              [0, 1]])
B = np.array([[1],
              [0.5]])
nx, nu = B.shape

print(f"\nSystem: x+ = Ax + Bu + w")
print(f"  States: {nx},  Inputs: {nu}")
print(f"  |x_i| <= {x_max},  |u| <= {u_max},  |w_i| <= {w_max}")

# ══════════════════════════════════════════════════════════════════
#  STEP 1: LQR feedback gain
# ══════════════════════════════════════════════════════════════════
K, Qf, _ = dlqr(A, B, Q_weight, R_weight)
K = -K                           # convention: u = Kx (negative feedback)
A_cl = A + B @ K                 # closed-loop dynamics

eigs = np.linalg.eigvals(A_cl)
print(f"\nLQR gain K = {K.flatten()}")
print(f"Closed-loop eigenvalues: {eigs}")
assert all(abs(e) < 1 for e in eigs), "Closed-loop is not stable!"

# ══════════════════════════════════════════════════════════════════
#  STEP 2: Constraint sets as polytopes
# ══════════════════════════════════════════════════════════════════
X = Polyhedron.from_Hrep(A=np.vstack((np.eye(nx), -np.eye(nx))),
                         b=x_max * np.ones(2 * nx))
U = Polyhedron.from_Hrep(A=np.vstack((np.eye(nu), -np.eye(nu))),
                         b=u_max * np.ones(2 * nu))
W = Polyhedron.from_Hrep(A=np.vstack((np.eye(nx), -np.eye(nx))),
                         b=w_max * np.ones(2 * nx))

print(f"\nConstraint sets constructed:")
print(f"  X: {X.A.shape[0]} half-planes")
print(f"  U: {U.A.shape[0]} half-planes")
print(f"  W: {W.A.shape[0]} half-planes")

# ══════════════════════════════════════════════════════════════════
#  STEP 3: Minimal robust positively invariant (mRPI) set
# ══════════════════════════════════════════════════════════════════
#  E = W ⊕ A_cl W ⊕ A_cl² W ⊕ ...
#  Stop when ||A_cl^i||₂ < tolerance (contributions become negligible)

def min_robust_invariant_set(A_cl, W, max_iter=30, tol=1e-2):
    """Compute the mRPI set via iterative Minkowski sums."""
    Omega = W
    for i in range(1, max_iter + 1):
        A_cl_i = np.linalg.matrix_power(A_cl, i)
        Omega_next = Omega + A_cl_i @ W       # Minkowski sum
        Omega_next.minHrep()

        if np.linalg.norm(A_cl_i, ord=2) < tol:
            print(f"  mRPI converged after {i} iterations "
                  f"(||A_cl^{i}||₂ = {np.linalg.norm(A_cl_i, ord=2):.2e})")
            return Omega_next

        Omega = Omega_next

    print(f"  WARNING: mRPI did NOT converge after {max_iter} iterations")
    return Omega

print("\nComputing mRPI set E ...")
E = min_robust_invariant_set(A_cl, W)
print(f"  E has {E.A.shape[0]} half-planes")

# ══════════════════════════════════════════════════════════════════
#  STEP 4: Constraint tightening (Pontryagin difference)
# ══════════════════════════════════════════════════════════════════
#  X̃ = X ⊖ E     (tightened state constraints)
#  Ũ = U ⊖ KE    (tightened input constraints)

print("\nTightening constraints ...")
X_tilde = X - E                           # Pontryagin difference
KE = E.affine_map(K)                      # image of E under K
U_tilde = U - KE                          # Pontryagin difference

print(f"  X̃ has {X_tilde.A.shape[0]} half-planes, "
      f"non-empty: {not X_tilde.is_empty()}")
print(f"  Ũ has {U_tilde.A.shape[0]} half-planes, "
      f"non-empty: {not U_tilde.is_empty()}")

# Verify using support function (alternative method)
U_tilde_b = U.b.copy()
for i in range(U_tilde_b.shape[0]):
    U_tilde_b[i] -= KE.support(U.A[i, :])
U_tilde_check = Polyhedron.from_Hrep(A=U.A, b=U_tilde_b)
print(f"  Support-function method matches: {U_tilde == U_tilde_check}")

# ══════════════════════════════════════════════════════════════════
#  STEP 5: Terminal sets (maximal invariant sets)
# ══════════════════════════════════════════════════════════════════

def max_invariant_set(A_cl, X, max_iter=30):
    """Compute the maximal invariant set for x+ = A_cl x inside X."""
    O = X
    for i in range(1, max_iter + 1):
        F, f = O.A, O.b
        O_next = Polyhedron.from_Hrep(
            np.vstack((F, F @ A_cl)),
            np.concatenate((f, f))
        )
        O_next.minHrep()
        if O_next == O:
            print(f"  Converged after {i} iterations")
            return O_next
        O = O_next
    print(f"  WARNING: did NOT converge after {max_iter} iterations")
    return O

# Terminal set for nominal MPC: Xf ⊆ X ∩ {x | Kx ∈ U}
print("\nComputing terminal set for nominal MPC ...")
X_and_KU = X.intersect(Polyhedron.from_Hrep(U.A @ K, U.b))
Xf = max_invariant_set(A_cl, X_and_KU)

# Terminal set for tube MPC: X̃f ⊆ X̃ ∩ {x | Kx ∈ Ũ}
print("Computing terminal set for tube MPC ...")
X_tilde_and_KU_tilde = X_tilde.intersect(
    Polyhedron.from_Hrep(U_tilde.A @ K, U_tilde.b)
)
Xf_tilde = max_invariant_set(A_cl, X_tilde_and_KU_tilde)

# ══════════════════════════════════════════════════════════════════
#  STEP 6: Tube MPC formulation (CVXPY)
# ══════════════════════════════════════════════════════════════════
print(f"\nFormulating Tube MPC (N = {N_horizon}) ...")

z_var = cp.Variable((N_horizon + 1, nx), name='z')   # nominal states
v_var = cp.Variable((N_horizon, nu), name='v')        # nominal inputs
x0_par = cp.Parameter((nx,), name='x0')               # measured state

# Cost: sum of stage costs + terminal cost
cost = 0
for k in range(N_horizon):
    cost += cp.quad_form(z_var[k], Q_weight)
    cost += cp.quad_form(v_var[k], R_weight)
cost += cp.quad_form(z_var[-1], Qf)

# Constraints
constraints = []

# Initial condition: x₀ ∈ z₀ ⊕ E  ⟺  E.A @ (x₀ - z₀) ≤ E.b
constraints.append(E.A @ (x0_par - z_var[0]) <= E.b)

# Nominal dynamics: z_{k+1} = A z_k + B v_k
constraints.append(z_var[1:].T == A @ z_var[:-1].T + B @ v_var.T)

# Tightened state constraints
constraints.append(X_tilde.A @ z_var[:-1].T <= X_tilde.b.reshape(-1, 1))

# Tightened input constraints
constraints.append(U_tilde.A @ v_var.T <= U_tilde.b.reshape(-1, 1))

# Terminal constraint: z_N ∈ X̃f
constraints.append(Xf_tilde.A @ z_var[-1] <= Xf_tilde.b)

tube_mpc = cp.Problem(cp.Minimize(cost), constraints)
print(f"  Problem size: {tube_mpc.size_metrics}")

# ══════════════════════════════════════════════════════════════════
#  STEP 7: Closed-loop simulation
# ══════════════════════════════════════════════════════════════════
print(f"\nRunning closed-loop simulation ({N_samples} samples, "
      f"{N_sim} steps each) ...")

# Sample initial state from the tightened terminal set
x0 = Xf_tilde.sample(1).flatten()
print(f"  Initial state: x₀ = [{x0[0]:.3f}, {x0[1]:.3f}]")

x_trajs = np.zeros((N_samples, N_sim + 1, nx))
u_trajs = np.zeros((N_samples, N_sim, nu))

for i in range(N_samples):
    x_trajs[i, 0] = x0
    xk = x0.copy()

    for k in range(N_sim):
        # Solve tube MPC
        x0_par.value = xk
        tube_mpc.solve(cp.PIQP)
        assert tube_mpc.status == cp.OPTIMAL, \
            f"Solver returned: {tube_mpc.status} at sample {i}, step {k}"

        z0 = z_var[0].value
        v0 = v_var[0].value

        # Tube control law: u = v₀* + K(x - z₀*)
        uk = v0 + K @ (xk - z0)

        # Random disturbance
        wk = W.sample(1).flatten()

        # True dynamics
        xk = A @ xk + B @ uk + wk

        x_trajs[i, k + 1] = xk.flatten()
        u_trajs[i, k] = uk.flatten()

print("  Simulation complete.")

# Check constraint satisfaction
x_viol = np.any(np.abs(x_trajs) > x_max + 1e-6)
u_viol = np.any(np.abs(u_trajs) > u_max + 1e-6)
print(f"  State constraint violations: {'YES' if x_viol else 'None'}")
print(f"  Input constraint violations: {'YES' if u_viol else 'None'}")

# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
print("\nGenerating plots ...")
t = np.arange(N_sim + 1)

# --- Figure 1: Polytope sets ---
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: X and E
ax = axes1[0]
X.plot(ax, color='green', opacity=0.3, label=r'$\mathcal{X}$')
E.plot(ax, color='red', opacity=0.5, label=r'$\mathcal{E}$ (mRPI)')
ax.set_xlabel(r'$x_1$ (position)')
ax.set_ylabel(r'$x_2$ (velocity)')
ax.set_title('State set and mRPI set')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Panel 2: X, X̃, E
ax = axes1[1]
X.plot(ax, color='green', opacity=0.2, label=r'$\mathcal{X}$')
X_tilde.plot(ax, color='gold', opacity=0.4,
             label=r'$\tilde{\mathcal{X}} = \mathcal{X} \ominus \mathcal{E}$')
E.plot(ax, color='red', opacity=0.5, label=r'$\mathcal{E}$')
ax.set_xlabel(r'$x_1$ (position)')
ax.set_ylabel(r'$x_2$ (velocity)')
ax.set_title('Constraint tightening')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Panel 3: Terminal sets
ax = axes1[2]
Xf.plot(ax, color='green', opacity=0.3,
        label=r'$\mathcal{X}_f$ (nominal)')
Xf_tilde.plot(ax, color='steelblue', opacity=0.5,
              label=r'$\tilde{\mathcal{X}}_f$ (tube)')
ax.set_xlabel(r'$x_1$ (position)')
ax.set_ylabel(r'$x_2$ (velocity)')
ax.set_title('Terminal sets')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

fig1.suptitle('Tube MPC — Polytope Design', fontsize=14, fontweight='bold')
fig1.tight_layout()

# --- Figure 2: Time trajectories ---
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

for x_traj, u_traj in zip(x_trajs, u_trajs):
    axes2[0].plot(t, x_traj[:, 0], linewidth=0.7, alpha=0.6)
    axes2[1].plot(t, x_traj[:, 1], linewidth=0.7, alpha=0.6)
    axes2[2].plot(t[:-1], u_traj[:, 0], linewidth=0.7, alpha=0.6)

# Constraint lines
for ax_idx in [0, 1]:
    axes2[ax_idx].axhline(x_max, color='red', linestyle='--',
                          linewidth=1.5, label='constraint')
    axes2[ax_idx].axhline(-x_max, color='red', linestyle='--',
                          linewidth=1.5)
axes2[2].axhline(u_max, color='red', linestyle='--',
                 linewidth=1.5, label='constraint')
axes2[2].axhline(-u_max, color='red', linestyle='--', linewidth=1.5)

axes2[0].set_ylabel(r'$x_1$ (position)')
axes2[1].set_ylabel(r'$x_2$ (velocity)')
axes2[2].set_ylabel(r'$u$ (input)')
axes2[2].set_xlabel('Time step $k$')

for ax in axes2:
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

fig2.suptitle(f'Closed-loop trajectories ({N_samples} realisations)',
              fontsize=14, fontweight='bold')
fig2.align_ylabels()
fig2.tight_layout()

# --- Figure 3: Phase portrait ---
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 6))
X.plot(ax3, opacity=0.1, color='green', label=r'$\mathcal{X}$')
X_tilde.plot(ax3, opacity=0.15, color='steelblue',
             label=r'$\tilde{\mathcal{X}}$')
Xf_tilde.plot(ax3, opacity=0.2, color='gold',
              label=r'$\tilde{\mathcal{X}}_f$')

for x_traj in x_trajs:
    ax3.plot(x_traj[:, 0], x_traj[:, 1], linewidth=0.7, alpha=0.5)
ax3.plot(x0[0], x0[1], 'ko', markersize=6, label=r'$x_0$')

ax3.set_xlabel(r'$x_1$ (position)')
ax3.set_ylabel(r'$x_2$ (velocity)')
ax3.set_title('Phase portrait — all trajectories stay inside '
              r'$\mathcal{X}$', fontsize=12)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')
fig3.tight_layout()

plt.show()

print("\n" + "=" * 60)
print("  Demo complete.")
print("=" * 60)
print("\nKey takeaways:")
print("  • mRPI set E captures worst-case disturbance accumulation")
print("  • Constraint tightening (X̃ = X ⊖ E) leaves room for the tube")
print("  • Tube law u = v* + K(x - z*) keeps real state inside the tube")
print("  • All constraints satisfied despite random disturbances")
