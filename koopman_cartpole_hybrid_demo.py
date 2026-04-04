"""
Koopman Cart-Pole Hybrid Demo
=============================

Standalone Python script for the cart-pole example used in the notes.

Method:
  1. Generate cart-pole data from the nonlinear simulator.
  2. Learn a lifted Koopman predictor with EDMD.
  3. Use energy-based swing-up away from upright.
  4. Use local Koopman MPC near upright, blended with a local stabilizer.

Requirements:
    pip install numpy matplotlib

Example:
    python koopman_cartpole_hybrid_demo.py --theta0_deg 180 --save koopman_cartpole_result.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Config:
    mc: float = 1.0
    mp: float = 0.3
    lp: float = 0.5
    g: float = 9.81
    ts: float = 0.02
    tc: float = 0.04
    t_final: float = 10.0
    u_max: float = 15.0
    theta0_deg: float = 180.0
    x0: float = 0.0
    xd0: float = 0.0
    thd0: float = 0.0
    seed: int = 12345
    n_traj: int = 200
    traj_len: int = 25
    n_mpc: int = 16
    save: str = "koopman_cartpole_result.png"
    no_show: bool = False


P_LIFT = 10


def wrap_angle(a: float) -> float:
    while a > np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return a


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def cartpole_deriv(s: np.ndarray, u: float, cfg: Config) -> np.ndarray:
    x, xd, th, thd = s
    sth = np.sin(th)
    cth = np.cos(th)
    mt = cfg.mc + cfg.mp
    denom = mt - cfg.mp * cth * cth
    thdd = (cfg.g * sth * mt - cth * (u + cfg.mp * cfg.lp * thd * thd * sth)) / (cfg.lp * denom)
    xdd = (u + cfg.mp * cfg.lp * (thd * thd * sth - thdd * cth)) / mt
    return np.array([xd, xdd, thd, thdd], dtype=float)


def rk4_step(s: np.ndarray, u: float, dt: float, cfg: Config) -> np.ndarray:
    k1 = cartpole_deriv(s, u, cfg)
    k2 = cartpole_deriv(s + 0.5 * dt * k1, u, cfg)
    k3 = cartpole_deriv(s + 0.5 * dt * k2, u, cfg)
    k4 = cartpole_deriv(s + dt * k3, u, cfg)
    sn = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    sn[2] = wrap_angle(float(sn[2]))
    return sn


def sim_step(s: np.ndarray, u: float, cfg: Config) -> np.ndarray:
    n_steps = int(round(cfg.tc / cfg.ts))
    sn = s.copy()
    for _ in range(n_steps):
        sn = rk4_step(sn, u, cfg.ts, cfg)
    sn[2] = wrap_angle(float(sn[2]))
    return sn


def lift_state(s: np.ndarray) -> np.ndarray:
    x, xd, th, thd = s
    sth = np.sin(th)
    cth = np.cos(th)
    return np.array(
        [x, xd, th, thd, sth, cth, xd * sth, xd * cth, thd * sth, thd * cth],
        dtype=float,
    )


def train_edmd(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    zx = []
    zy = []
    for _ in range(cfg.n_traj):
        s = np.array(
            [
                2.0 * (rng.random() - 0.5),
                2.0 * (rng.random() - 0.5),
                2.0 * np.pi * (rng.random() - 0.5),
                4.0 * (rng.random() - 0.5),
            ],
            dtype=float,
        )
        for _ in range(cfg.traj_len):
            u = 2.0 * cfg.u_max * (rng.random() - 0.5)
            z_now = lift_state(s)
            s_next = sim_step(s, u, cfg)
            z_next = lift_state(s_next)
            zx.append(np.concatenate([z_now, [u]]))
            zy.append(z_next)
            s = s_next

    zx_arr = np.asarray(zx)
    zy_arr = np.asarray(zy)
    xtx = zx_arr.T @ zx_arr + 1e-6 * np.eye(P_LIFT + 1)
    xty = zx_arr.T @ zy_arr
    w = np.linalg.solve(xtx, xty)
    a_k = w[:P_LIFT, :]
    b_k = w[P_LIFT, :]
    return a_k, b_k


def build_affine_lifted_qp(
    a_k: np.ndarray,
    b_k: np.ndarray,
    e0: np.ndarray,
    c_off: np.ndarray,
    horizon: int,
    q_diag: np.ndarray,
    r_val: float,
    qf_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    apow = [np.eye(P_LIFT)]
    for _ in range(horizon):
        apow.append(a_k @ apow[-1])

    gb = [ap @ b_k for ap in apow]

    free = [e0.copy()]
    cur = e0.copy()
    for _ in range(horizon):
        cur = a_k @ cur + c_off
        free.append(cur.copy())

    h = np.zeros((horizon, horizon), dtype=float)
    f = np.zeros(horizon, dtype=float)

    for i in range(horizon):
        for j in range(i, horizon):
            hs = 0.0
            for k in range(j + 1, horizon + 1):
                w = q_diag if k < horizon else qf_diag
                hs += np.sum(gb[k - i - 1] * w * gb[k - j - 1])
            if i == j:
                hs += r_val
            h[i, j] = hs
            h[j, i] = hs

        fs = 0.0
        for k in range(i + 1, horizon + 1):
            w = q_diag if k < horizon else qf_diag
            fs += np.sum(gb[k - i - 1] * w * free[k])
        f[i] = fs

    return h, f


def solve_box_qp(h: np.ndarray, f: np.ndarray, u_max: float, max_iter: int = 800) -> np.ndarray:
    n = f.shape[0]
    u = np.zeros(n, dtype=float)
    alpha = 1.0 / (np.max(np.diag(h)) + 1e-8)
    for _ in range(max_iter):
        grad = h @ u + f
        u_new = np.clip(u - alpha * grad, -u_max, u_max)
        if np.max(np.abs(u_new - u)) < 1e-8:
            u = u_new
            break
        u = u_new
    return u


def linearize_cartpole_step(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.zeros(4)
    eps = 1e-6
    a = np.zeros((4, 4), dtype=float)
    b = np.zeros(4, dtype=float)
    for j in range(4):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        fp = sim_step(xp, 0.0, cfg)
        fm = sim_step(xm, 0.0, cfg)
        a[:, j] = (fp - fm) / (2.0 * eps)
    fp = sim_step(x0.copy(), eps, cfg)
    fm = sim_step(x0.copy(), -eps, cfg)
    b[:] = (fp - fm) / (2.0 * eps)
    return a, b


def solve_dare_iterative(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: float) -> np.ndarray:
    p = q.copy()
    bcol = b.reshape(-1, 1)
    for _ in range(1000):
        bt_p = bcol.T @ p
        s = r + float((bt_p @ bcol).item())
        p_next = q + a.T @ p @ a - (a.T @ p @ bcol @ bt_p @ a) / s
        if np.max(np.abs(p_next - p)) < 1e-11:
            p = p_next
            break
        p = p_next
    gain = np.linalg.solve(np.array([[r + float((bcol.T @ p @ bcol).item())]]), bcol.T @ p @ a)
    return np.asarray(gain).reshape(-1)


def create_controller(a_k: np.ndarray, b_k: np.ndarray, cfg: Config):
    z_ref = lift_state(np.zeros(4))
    c_off = a_k @ z_ref - z_ref

    q_diag = np.zeros(P_LIFT, dtype=float)
    q_diag[:6] = np.array([18.0, 2.0, 22.0, 3.0, 18.0, 10.0])
    qf_diag = 6.0 * q_diag

    a_lin, b_lin = linearize_cartpole_step(cfg)
    q_local = np.diag([15.0, 2.0, 120.0, 12.0])
    k_local = solve_dare_iterative(a_lin, b_lin, q_local, 0.3)

    def local_control(state: np.ndarray) -> float:
        e = np.array([state[0], state[1], wrap_angle(float(state[2])), state[3]])
        return clamp(float(-k_local @ e), -cfg.u_max, cfg.u_max)

    def koopman_mpc_control(state: np.ndarray) -> float:
        z = lift_state(state)
        e0 = z - z_ref
        h, f = build_affine_lifted_qp(a_k, b_k, e0, c_off, cfg.n_mpc, q_diag, 0.2, qf_diag)
        return float(solve_box_qp(h, f, cfg.u_max, 800)[0])

    def swing_up_control(state: np.ndarray) -> float:
        x, xd, th, thd = state
        inertia = cfg.mp * cfg.lp * cfg.lp
        energy = 0.5 * inertia * thd * thd + cfg.mp * cfg.g * cfg.lp * (np.cos(th) - 1.0)
        swing = 35.0 * energy * np.sign(thd * np.cos(th) + 1e-4)
        center = -1.0 * x - 2.0 * xd
        return clamp(float(swing + center), -cfg.u_max, cfg.u_max)

    def controller(state: np.ndarray) -> float:
        th = abs(wrap_angle(float(state[2])))
        if th < 0.35 and abs(float(state[3])) < 2.0:
            u_mpc = koopman_mpc_control(state)
            u_local = local_control(state)
            return clamp(0.15 * u_mpc + 0.85 * u_local, -cfg.u_max, cfg.u_max)
        return swing_up_control(state)

    return controller


def simulate(controller, cfg: Config, s0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sim = int(round(cfg.t_final / cfg.tc))
    states = [s0.copy()]
    controls = []
    times = [0.0]
    s = s0.copy()
    for k in range(n_sim):
        s[2] = wrap_angle(float(s[2]))
        u = clamp(float(controller(s)), -cfg.u_max, cfg.u_max)
        controls.append(u)
        s = sim_step(s, u, cfg)
        states.append(s.copy())
        times.append((k + 1) * cfg.tc)
    return np.asarray(times), np.asarray(states), np.asarray(controls)


def make_plots(times: np.ndarray, states: np.ndarray, controls: np.ndarray, cfg: Config) -> None:
    theta_deg = np.degrees(np.vectorize(wrap_angle)(states[:, 2]))
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(times, states[:, 0], color="#e74c3c", lw=2)
    axes[0].axhline(0.0, color="0.75", ls="--", lw=1)
    axes[0].set_ylabel("x (m)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(times, states[:, 1], color="#9b59b6", lw=2)
    axes[1].axhline(0.0, color="0.75", ls="--", lw=1)
    axes[1].set_ylabel("xdot (m/s)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(times, theta_deg, color="#3498db", lw=2)
    axes[2].axhline(0.0, color="0.75", ls="--", lw=1)
    axes[2].set_ylabel("theta (deg)")
    axes[2].grid(alpha=0.25)

    axes[3].step(times[:-1], controls, where="post", color="#f39c12", lw=2)
    axes[3].axhline(cfg.u_max, color="0.8", ls="--", lw=1)
    axes[3].axhline(-cfg.u_max, color="0.8", ls="--", lw=1)
    axes[3].set_ylabel("u (N)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=0.25)

    fig.suptitle("Cart-Pole Hybrid Swing-Up with Koopman MPC", fontsize=14)
    fig.tight_layout()
    out_path = Path(cfg.save)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    if not cfg.no_show:
        plt.show()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Koopman cart-pole hybrid demo")
    parser.add_argument("--theta0_deg", type=float, default=180.0)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--xd0", type=float, default=0.0)
    parser.add_argument("--thd0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n_traj", type=int, default=200)
    parser.add_argument("--traj_len", type=int, default=25)
    parser.add_argument("--n_mpc", type=int, default=16)
    parser.add_argument("--save", type=str, default="koopman_cartpole_result.png")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()
    return Config(
        theta0_deg=args.theta0_deg,
        x0=args.x0,
        xd0=args.xd0,
        thd0=args.thd0,
        seed=args.seed,
        n_traj=args.n_traj,
        traj_len=args.traj_len,
        n_mpc=args.n_mpc,
        save=args.save,
        no_show=args.no_show,
    )


def main() -> None:
    cfg = parse_args()
    print("Training EDMD model...")
    a_k, b_k = train_edmd(cfg)
    controller = create_controller(a_k, b_k, cfg)
    s0 = np.array([cfg.x0, cfg.xd0, np.deg2rad(cfg.theta0_deg), cfg.thd0], dtype=float)
    times, states, controls = simulate(controller, cfg, s0)
    print("Final state:", states[-1])
    print("Max |x|:", float(np.max(np.abs(states[:, 0]))))
    make_plots(times, states, controls, cfg)


if __name__ == "__main__":
    main()
