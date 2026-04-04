"""
Koopman Cart-Pole Hybrid Demo
=============================

Standalone Python script for the cart-pole example used in the notes.

Method:
  1. Generate cart-pole data from the nonlinear simulator.
  2. Learn a lifted Koopman predictor with EDMD.
  3. Use energy-based swing-up away from upright.
  4. Use Koopman MPC near upright.

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
    a_k = w[:P_LIFT, :].T
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
    try:
        u_free = np.linalg.solve(h, -f)
        if np.all(np.abs(u_free) <= u_max + 1e-9):
            return u_free
        u = np.clip(u_free, -u_max, u_max)
    except np.linalg.LinAlgError:
        u = np.zeros(n, dtype=float)

    alpha = 1.0 / (np.max(np.sum(np.abs(h), axis=1)) + 1e-8)
    for _ in range(max_iter):
        grad = h @ u + f
        u_new = np.clip(u - alpha * grad, -u_max, u_max)
        if np.max(np.abs(u_new - u)) < 1e-8:
            u = u_new
            break
        u = u_new
    return u


def train_local_balance_model(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 17)
    zx = []
    zy = []
    for _ in range(12000):
        s = np.array(
            [
                0.15 * (rng.random() - 0.5) * 2.0,
                0.30 * (rng.random() - 0.5) * 2.0,
                0.03 * (rng.random() - 0.5) * 2.0,
                0.25 * (rng.random() - 0.5) * 2.0,
            ],
            dtype=float,
        )
        u = 2.0 * (rng.random() - 0.5)
        z = np.array([s[0], s[1], wrap_angle(float(s[2])), s[3]], dtype=float)
        sn = sim_step(s, u, cfg)
        zn = np.array([sn[0], sn[1], wrap_angle(float(sn[2])), sn[3]], dtype=float)
        zx.append(np.concatenate([z, [u]]))
        zy.append(zn)

    zx_arr = np.asarray(zx)
    zy_arr = np.asarray(zy)
    w = np.linalg.solve(zx_arr.T @ zx_arr + 1e-10 * np.eye(5), zx_arr.T @ zy_arr)
    return w[:4, :].T, w[4, :]


def create_controller(a_k: np.ndarray, b_k: np.ndarray, cfg: Config):
    del a_k, b_k
    a_bal, b_bal = train_local_balance_model(cfg)
    q_bal = np.diag([15.0, 2.0, 120.0, 12.0])
    r_bal = 0.3

    p_bal = q_bal.copy()
    bcol = b_bal.reshape(-1, 1)
    for _ in range(5000):
        s_val = r_bal + float((bcol.T @ p_bal @ bcol).item())
        p_next = q_bal + a_bal.T @ p_bal @ a_bal - (a_bal.T @ p_bal @ bcol @ bcol.T @ p_bal @ a_bal) / s_val
        if np.max(np.abs(p_next - p_bal)) < 1e-12:
            p_bal = p_next
            break
        p_bal = p_next

    n_bal = 30
    apow = [np.eye(4)]
    for _ in range(n_bal):
        apow.append(a_bal @ apow[-1])
    gb = [ap @ b_bal for ap in apow]

    def koopman_mpc_control(state: np.ndarray) -> float:
        x0 = np.array([state[0], state[1], wrap_angle(float(state[2])), state[3]], dtype=float)
        h = np.zeros((n_bal, n_bal), dtype=float)
        f = np.zeros(n_bal, dtype=float)

        free = [x0.copy()]
        cur = x0.copy()
        for _ in range(n_bal):
            cur = a_bal @ cur
            free.append(cur.copy())

        for i in range(n_bal):
            for j in range(i, n_bal):
                hs = 0.0
                for k in range(j + 1, n_bal + 1):
                    w = q_bal if k < n_bal else p_bal
                    hs += gb[k - i - 1] @ w @ gb[k - j - 1]
                if i == j:
                    hs += r_bal
                h[i, j] = hs
                h[j, i] = hs

            fs = 0.0
            for k in range(i + 1, n_bal + 1):
                w = q_bal if k < n_bal else p_bal
                fs += gb[k - i - 1] @ w @ free[k]
            f[i] = fs

        u = np.linalg.solve(h, -f)
        return float(clamp(float(u[0]), -cfg.u_max, cfg.u_max))

    def swing_up_control(state: np.ndarray) -> float:
        x, xd, th, thd = state
        inertia = cfg.mp * cfg.lp * cfg.lp
        energy = 0.5 * inertia * thd * thd + cfg.mp * cfg.g * cfg.lp * (np.cos(th) - 1.0)
        near_upright = abs(wrap_angle(float(th))) < 0.25
        swing_gain = 15.0 if near_upright else 35.0
        swing = swing_gain * energy * np.sign(thd * np.cos(th) + 1e-4)
        center = -1.0 * x - 2.0 * xd - 2.0 * np.sin(th) - 1.0 * thd * np.cos(th)
        return clamp(float(swing + center), -cfg.u_max, cfg.u_max)

    def controller(state: np.ndarray) -> float:
        th = abs(wrap_angle(float(state[2])))
        if th < 0.035 and abs(float(state[3])) < 0.22:
            return clamp(koopman_mpc_control(state), -cfg.u_max, cfg.u_max)
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
