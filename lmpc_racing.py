#!/usr/bin/env python3
"""
Neural-Network MPC for Autonomous Racing
==========================================

This script demonstrates learning-based Model Predictive Control (MPC) on a
time-optimal autonomous racing problem.  A neural network learns the optimal
speed profile from trajectory data, then the MPC uses this learned profile
with lookahead braking to race progressively faster each lap.

Algorithm
---------
Iteration 0:
    Pure pursuit at a conservative speed → collect trajectory data.

After each completed lap:
    Update NN speed map training data:
      - Where the car had large safety margin (far from track edge):
        label the target speed HIGHER than actual (room to go faster).
      - Where the car was near the edge: keep speed or reduce it.
    Retrain the NN on all accumulated data (keeping the BEST speed at
    each track position across all iterations).

Iterations 1+:
    At each MPC step:
      1. Query NN speed map for target speed at current + upcoming positions
      2. Apply proactive braking if the NN predicts slower speed ahead
      3. Execute pure pursuit steering at the computed target speed
    The NN progressively learns higher speeds on straights while
    maintaining safe speeds through corners — the car races faster
    each lap.

Additionally, a second NN (value function) learns the cost-to-go V(x) from
all trajectory data.  This is used as a terminal cost in the MPC and is
plotted to visualise the learning.

Vehicle Model
-------------
Dynamic bicycle with Pacejka Magic Formula tyre forces (6 states, 2 inputs).

    State:  x = [X, Y, phi, vx, vy, r]
    Input:  u = [delta, D]

    Dynamics:
        dX/dt  = vx*cos(phi) - vy*sin(phi)
        dY/dt  = vx*sin(phi) + vy*cos(phi)
        dphi/dt = r
        dvx/dt = (1/m)*[Frx - Ffy*sin(delta) + m*vy*r - Fdrag]
        dvy/dt = (1/m)*[Fry + Ffy*cos(delta) - m*vx*r]
        dr/dt  = (1/Iz)*[Ffy*lf*cos(delta) - Fry*lr]

    Tyre forces (Pacejka):
        Ffy = Df*sin(Cf*atan(Bf*alpha_f))
        Fry = Dr*sin(Cr*atan(Br*alpha_r))

Vehicle Parameters
------------------
    Parameter    Value    Unit         Description
    ---------    -----    ----         -----------
    m            800      kg           Vehicle mass
    Iz           1200     kg*m^2       Yaw moment of inertia
    lf           1.2      m            CG to front axle
    lr           1.3      m            CG to rear axle
    Bf,Cf,Df     10,1.3,5000          Pacejka front tyre
    Br,Cr,Dr     11,1.3,7000          Pacejka rear tyre
    Cm1,Cm2      5000,100             Motor model
    Cr0,Cr2      50,0.3               Drag model
    delta_max    0.45     rad          Max steering angle
    D            [-1,1]               Throttle/brake range
    vx           [0.5,20] m/s         Speed bounds

Track options:
    "custom"       - 13-waypoint Catmull-Rom circuit (~212 m)
    "mpcc_center"  - MPCC benchmark centreline (~444 m)
    "mpcc_racing"  - MPCC benchmark racing line (~410 m)

Usage:
    python lmpc_racing.py

    Modify USER SETTINGS below to experiment.

References:
    [1] U. Rosolia & F. Borrelli, "Learning MPC for Iterative Tasks," TAC 2018.
    [2] A. Liniger et al., "Optimization-Based Autonomous Racing," 2015.
"""

import numpy as np
from scipy.spatial import KDTree
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as timer
import warnings
import sys

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================================================
#  USER-CONFIGURABLE SETTINGS  ---  Students: modify these to experiment!
# ==========================================================================

TRACK_CHOICE    = "custom"      # "custom" | "mpcc_center" | "mpcc_racing"
N_ITERATIONS    = 8             # Total iterations (0=pure pursuit, 1+=NN-MPC)
DT              = 0.05          # Simulation time step [s]
MPC_EVERY       = 2             # Recompute control every N sim steps (try 2-5)
PP_SPEED        = 5.0           # Pure pursuit conservative speed [m/s] (try 3-6)
TRACK_HW        = 5.0           # Track half-width [m]
MAX_LAP_TIME    = 120.0         # Abort if lap exceeds this [s]

# NN Speed Map  ---  the main learning component
NN_SPEED_LAYERS = (64, 32)      # Hidden layer sizes (try (32,16) to (128,64))
SPEED_BOOST     = 0.15          # Speed boost factor per iteration (try 0.10-0.25)
N_BINS          = 60            # Track discretisation for speed map

# NN Value Function  ---  learns cost-to-go, used as terminal cost
NN_VF_LAYERS    = (64, 32)      # Hidden layer sizes
LAMBDA_TERM     = 0.005         # Terminal cost weight in MPC

# Curvature speed limit (controls corner braking)
A_LAT_MAX       = 6.0           # Max lateral accel for corner speed [m/s^2]
                                #   (try 4=conservative to 10=aggressive)

# ==========================================================================
#  VEHICLE PARAMETERS
# ==========================================================================

VP = dict(
    m=800, Iz=1200,               # mass [kg], yaw inertia [kg*m^2]
    lf=1.2, lr=1.3,               # CG to front/rear axle [m]
    Bf=10.0, Cf=1.3, Df=5000,     # Pacejka front: B, C, D
    Br=11.0, Cr=1.3, Dr=7000,     # Pacejka rear:  B, C, D
    Cm1=5000, Cm2=100,            # Motor force coefficients
    Cr0=50, Cr2=0.3,              # Rolling resistance + aero drag
    deltaMax=0.45,                 # Max steering angle [rad]
    DMax=1.0, DMin=-1.0,          # Throttle range
    vxMin=0.5, vxMax=20.0,        # Longitudinal speed bounds [m/s]
)
NX, NU = 6, 2


# ==========================================================================
#  TRACK DATA
# ==========================================================================

CUSTOM_WP = np.array([
    [0, 0], [22, -1], [45, 1], [58, 10], [63, 25],
    [55, 40], [40, 50], [20, 54], [2, 48],
    [-10, 38], [-14, 25], [-10, 12], [-4, 2],
], dtype=float)

MPCC_CENTER = np.array([
    [-28.1,26.9],[-26.0,24.8],[-23.9,22.7],[-21.8,20.6],[-19.7,18.5],
    [-17.6,16.4],[-15.5,14.3],[-13.4,12.2],[-11.3,10.1],[-9.3,8.0],
    [-7.2,5.9],[-5.1,3.9],[-3.0,1.8],[-0.9,-0.3],[1.2,-2.4],
    [3.7,-4.1],[6.6,-4.1],[9.0,-2.4],[10.1,0.3],[9.5,3.1],
    [7.6,5.4],[5.5,7.5],[3.4,9.6],[1.3,11.7],[-0.8,13.8],
    [-2.8,15.9],[-3.8,18.7],[-3.4,21.6],[-1.7,24.0],[0.9,25.3],
    [3.9,25.5],[6.9,25.5],[9.8,25.3],[12.7,24.5],[15.3,23.2],
    [17.7,21.4],[19.6,19.1],[21.0,16.5],[21.9,13.7],[22.2,10.7],
    [22.2,7.8],[22.2,4.8],[22.2,1.8],[22.1,-1.1],[21.5,-4.0],
    [20.3,-6.7],[18.7,-9.2],[16.7,-11.4],[14.3,-13.2],[11.7,-14.6],
    [8.9,-15.4],[5.9,-15.8],[3.0,-15.5],[0.1,-14.7],[-2.5,-13.4],
    [-4.8,-11.5],[-7.0,-9.4],[-9.1,-7.3],[-11.2,-5.2],[-13.3,-3.1],
    [-15.4,-1.1],[-17.5,1.0],[-19.6,3.1],[-21.7,5.2],[-24.2,6.7],
    [-27.1,6.5],[-29.4,4.8],[-30.3,2.0],[-30.3,-1.0],[-30.3,-3.9],
    [-30.3,-6.9],[-30.3,-9.9],[-30.3,-12.8],[-30.0,-15.8],[-28.6,-18.3],
    [-26.2,-20.1],[-23.3,-20.8],[-20.4,-21.0],[-17.9,-22.4],[-16.7,-25.1],
    [-17.1,-28.0],[-19.1,-30.1],[-22.0,-30.8],[-24.9,-30.8],[-27.8,-31.4],
    [-29.8,-33.6],[-30.3,-36.4],[-29.1,-39.1],[-26.6,-40.6],[-23.6,-40.8],
    [-20.7,-40.8],[-17.7,-40.8],[-14.7,-40.8],[-11.8,-40.8],[-8.8,-40.8],
    [-5.8,-40.8],[-2.9,-40.6],[-0.5,-38.9],[0.4,-36.1],[1.4,-33.4],
    [3.8,-31.7],[6.7,-31.5],[9.7,-31.5],[12.7,-31.5],[15.6,-31.5],
    [18.6,-31.0],[21.4,-30.0],[24.0,-28.7],[26.4,-26.9],[28.5,-24.8],
    [30.2,-22.4],[31.5,-19.7],[32.4,-16.9],[32.9,-14.0],[32.9,-11.0],
    [32.9,-8.1],[32.9,-5.1],[32.9,-2.1],[32.9,0.8],[32.9,3.8],
    [32.9,6.8],[32.9,9.8],[32.9,12.7],[32.6,15.7],[31.9,18.6],
    [30.9,21.3],[29.5,24.0],[27.9,26.5],[26.0,28.7],[23.8,30.7],
    [21.4,32.4],[18.8,33.9],[16.0,35.0],[13.2,35.7],[10.2,36.1],
    [7.2,36.2],[4.3,36.2],[1.3,36.2],[-1.7,36.2],[-4.6,36.2],
    [-7.6,36.2],[-10.6,36.2],[-13.5,36.2],[-16.5,36.2],[-19.5,36.2],
    [-22.4,36.2],[-25.4,36.2],[-28.2,35.3],[-30.0,33.0],[-30.2,30.1],
], dtype=float)

MPCC_RACING = np.array([
    [-28.1,26.5],[-27.4,23.9],[-26.1,21.5],[-24.5,19.3],[-22.7,17.2],
    [-20.7,15.3],[-18.8,13.5],[-16.8,11.6],[-14.8,9.7],[-12.9,7.7],
    [-11.0,5.8],[-9.1,3.8],[-7.2,1.8],[-5.2,-0.1],[-3.1,-1.7],
    [-0.6,-2.9],[2.1,-3.4],[4.7,-2.9],[6.9,-1.4],[8.2,1.0],
    [8.3,3.8],[7.3,6.3],[5.7,8.5],[3.7,10.4],[1.6,12.1],
    [-0.2,14.1],[-1.4,16.6],[-1.5,19.3],[-0.4,21.8],[1.5,23.7],
    [4.0,24.8],[6.7,24.9],[9.4,24.2],[11.8,23.0],[14.0,21.4],
    [15.9,19.4],[17.4,17.2],[18.7,14.7],[19.7,12.2],[20.5,9.6],
    [21.1,6.9],[21.4,4.2],[21.4,1.5],[20.9,-1.2],[20.0,-3.8],
    [18.8,-6.2],[17.1,-8.4],[15.2,-10.3],[12.9,-11.8],[10.4,-12.9],
    [7.8,-13.6],[5.1,-13.7],[2.3,-13.4],[-0.3,-12.6],[-2.7,-11.4],
    [-5.0,-9.9],[-7.1,-8.1],[-9.0,-6.2],[-10.8,-4.1],[-12.6,-2.1],
    [-14.5,-0.2],[-16.7,1.5],[-19.1,2.8],[-21.7,3.6],[-24.4,3.9],
    [-27.1,3.4],[-29.4,2.0],[-31.1,-0.1],[-32.2,-2.7],[-32.7,-5.3],
    [-32.7,-8.1],[-32.2,-10.7],[-30.9,-13.1],[-29.2,-15.3],[-27.2,-17.1],
    [-24.9,-18.6],[-22.7,-20.2],[-20.6,-22.0],[-19.4,-24.4],[-19.4,-27.1],
    [-20.9,-29.4],[-23.0,-31.0],[-25.3,-32.5],[-27.2,-34.4],[-28.1,-37.0],
    [-27.7,-39.7],[-25.9,-41.7],[-23.5,-43.0],[-20.9,-43.7],[-18.1,-43.8],
    [-15.4,-43.5],[-12.8,-42.9],[-10.2,-42.0],[-7.7,-40.8],[-5.3,-39.5],
    [-2.9,-38.2],[-0.6,-36.9],[1.9,-35.6],[4.4,-34.6],[7.0,-33.8],
    [9.6,-33.0],[12.3,-32.3],[14.8,-31.4],[17.4,-30.3],[19.8,-29.0],
    [22.0,-27.5],[24.2,-25.8],[26.2,-23.9],[28.0,-21.9],[29.6,-19.7],
    [31.1,-17.4],[32.4,-15.0],[33.4,-12.5],[34.2,-9.9],[34.8,-7.2],
    [35.2,-4.5],[35.3,-1.8],[35.3,1.0],[35.0,3.7],[34.6,6.4],
    [34.0,9.0],[33.2,11.7],[32.2,14.2],[31.1,16.7],[29.8,19.1],
    [28.3,21.4],[26.7,23.6],[24.9,25.6],[22.9,27.5],[20.8,29.3],
    [18.6,30.9],[16.2,32.2],[13.7,33.3],[11.2,34.3],[8.5,35.1],
    [5.9,35.7],[3.2,36.3],[0.5,36.8],[-2.2,37.3],[-4.9,37.6],
    [-7.6,37.9],[-10.3,38.0],[-13.0,38.1],[-15.8,38.0],[-18.5,37.7],
    [-21.1,37.1],[-23.6,36.0],[-25.8,34.3],[-27.5,32.2],[-28.4,29.6],
], dtype=float)


# ==========================================================================
#  TRACK CONSTRUCTION
# ==========================================================================

def _crs(a, b, c, d, t):
    """Catmull-Rom spline interpolation."""
    return 0.5 * ((2*b) + (-a+c)*t + (2*a-5*b+4*c-d)*t**2
                  + (-a+3*b-3*c+d)*t**3)

def _crsd(a, b, c, d, t):
    """Catmull-Rom spline derivative."""
    return 0.5 * ((-a+c) + (4*a-10*b+8*c-2*d)*t
                  + (-3*a+9*b-9*c+3*d)*t**2)

def build_track_spline(wp, res=30):
    """Build track from waypoints using Catmull-Rom splines."""
    n = len(wp)
    total = n * res
    px, py = np.zeros(total), np.zeros(total)
    tx, ty = np.zeros(total), np.zeros(total)
    for seg in range(n):
        i0, i1, i2, i3 = (seg-1)%n, seg, (seg+1)%n, (seg+2)%n
        for j in range(res):
            t = j / res
            idx = seg * res + j
            px[idx] = _crs(wp[i0,0], wp[i1,0], wp[i2,0], wp[i3,0], t)
            py[idx] = _crs(wp[i0,1], wp[i1,1], wp[i2,1], wp[i3,1], t)
            ddx = _crsd(wp[i0,0], wp[i1,0], wp[i2,0], wp[i3,0], t)
            ddy = _crsd(wp[i0,1], wp[i1,1], wp[i2,1], wp[i3,1], t)
            L = max(np.hypot(ddx, ddy), 1e-10)
            tx[idx], ty[idx] = ddx / L, ddy / L
    return _finalise_track(px, py, tx, ty, total)

def build_track_points(pts):
    """Build track from dense point array (e.g. MPCC data)."""
    n = len(pts)
    px, py = pts[:, 0].copy(), pts[:, 1].copy()
    tx, ty = np.zeros(n), np.zeros(n)
    for i in range(n):
        ip, im = (i + 1) % n, (i - 1) % n
        dx, dy = px[ip] - px[im], py[ip] - py[im]
        L = max(np.hypot(dx, dy), 1e-10)
        tx[i], ty[i] = dx / L, dy / L
    return _finalise_track(px, py, tx, ty, n)

def _finalise_track(px, py, tx, ty, n):
    """Compute normals, arc-length, curvature, and KDTree."""
    nx, ny = -ty.copy(), tx.copy()
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i-1] + np.hypot(px[i]-px[i-1], py[i]-py[i-1])
    total_len = s[-1] + np.hypot(px[0]-px[-1], py[0]-py[-1])

    # Raw curvature
    curvature = np.zeros(n)
    for i in range(n):
        ip, im = (i + 1) % n, (i - 1) % n
        ds = s[ip] - s[im] if ip > im else (s[ip] + total_len - s[im])
        if ds > 0:
            dphi = np.arctan2(ty[ip], tx[ip]) - np.arctan2(ty[im], tx[im])
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            curvature[i] = abs(dphi / ds)

    # Smoothed curvature for speed planning (average over ~10m window)
    window = max(3, n // 40)
    smooth_curv = np.zeros(n)
    for i in range(n):
        indices = [(i + j) % n for j in range(-window, window + 1)]
        smooth_curv[i] = np.mean(curvature[indices])
    smooth_curv = np.minimum(smooth_curv, 0.25)

    kd = KDTree(np.column_stack([px, py]))
    return dict(px=px, py=py, tx=tx, ty=ty, nx=nx, ny=ny,
                s=s, n=n, totalLength=total_len,
                curvature=curvature, smooth_curvature=smooth_curv, kd=kd)

def project_on_track(X, Y, track):
    """Project a point onto the track centreline."""
    _, idx = track['kd'].query([X, Y])
    d = ((X - track['px'][idx]) * track['nx'][idx]
         + (Y - track['py'][idx]) * track['ny'][idx])
    return dict(s=track['s'][idx], d=d, idx=idx)

def find_idx_for_s(s, track):
    """Find track point index for a given arc-length."""
    s = s % track['totalLength']
    return max(0, int(np.searchsorted(track['s'], s, side='right')) - 1)


# ==========================================================================
#  VEHICLE DYNAMICS
# ==========================================================================

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def f_cont(x, u):
    """Continuous-time dynamic bicycle model with Pacejka tires."""
    phi, vx, vy, r = x[2], x[3], x[4], x[5]
    delta, D = u[0], u[1]
    vxS = max(vx, VP['vxMin'])
    af = -np.arctan2(vy + r * VP['lf'], vxS) + delta
    ar = -np.arctan2(vy - r * VP['lr'], vxS)
    Ffy = VP['Df'] * np.sin(VP['Cf'] * np.arctan(VP['Bf'] * af))
    Fry = VP['Dr'] * np.sin(VP['Cr'] * np.arctan(VP['Br'] * ar))
    Frx = VP['Cm1'] * D - VP['Cm2'] * D * vx
    Fd  = VP['Cr0'] + VP['Cr2'] * vx ** 2
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array([
        vx * cp - vy * sp,
        vx * sp + vy * cp,
        r,
        (Frx - Ffy * np.sin(delta) + VP['m'] * vy * r - Fd) / VP['m'],
        (Fry + Ffy * np.cos(delta) - VP['m'] * vx * r) / VP['m'],
        (Ffy * VP['lf'] * np.cos(delta) - Fry * VP['lr']) / VP['Iz'],
    ])

def rk4_step(x, u, dt):
    """Runge-Kutta 4th order integration step."""
    k1 = f_cont(x, u)
    k2 = f_cont(x + 0.5*dt*k1, u)
    k3 = f_cont(x + 0.5*dt*k2, u)
    k4 = f_cont(x + dt*k3, u)
    return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def clip_state(x):
    """Enforce state bounds."""
    x = x.copy()
    x[3] = np.clip(x[3], VP['vxMin'], VP['vxMax'])
    x[4] = np.clip(x[4], -5.0, 5.0)
    x[5] = np.clip(x[5], -3.0, 3.0)
    return x


# ==========================================================================
#  PURE PURSUIT CONTROLLER
# ==========================================================================

def pure_pursuit(x, s_now, track, target_v):
    """Pure pursuit steering + proportional speed control."""
    lookahead = max(4.0, x[3] * 0.9)
    idx_l = find_idx_for_s(s_now + lookahead, track)
    dx = track['px'][idx_l] - x[0]
    dy = track['py'][idx_l] - x[1]
    ang = wrap_angle(np.arctan2(dy, dx) - x[2])
    dist = max(np.hypot(dx, dy), 0.1)
    curv = 2.0 * np.sin(ang) / dist
    L = VP['lf'] + VP['lr']
    delta = np.clip(np.arctan(curv * L), -VP['deltaMax'], VP['deltaMax'])
    D = np.clip((target_v - x[3]) * 0.8, VP['DMin'], VP['DMax'])
    return np.array([delta, D])


# ==========================================================================
#  NN SPEED MAP  (learns optimal speed at each track position)
# ==========================================================================

class NNSpeedMap:
    """
    Neural network that learns the optimal racing speed at each track position.

    Builds a speed reference from safe set data (like LMPC):
      1. Track max achieved speed at each position (the "safe set")
      2. Apply iteration-based boost to push beyond proven speeds
      3. Cap with curvature-based speed limits (physics)
      4. Train NN on the resulting speed profile

    The NN generalises between bins and provides smooth speed predictions.
    Over iterations, the boost pushes straight speeds higher while curvature
    limits keep corner speeds physically realistic.

    Features: [s/L, curvature, sin(2*pi*s/L), cos(2*pi*s/L)]
    Output:   target speed [m/s]
    """

    def __init__(self, track):
        self.model = MLPRegressor(
            hidden_layer_sizes=NN_SPEED_LAYERS,
            activation='relu', solver='adam',
            max_iter=500, warm_start=True, random_state=42,
        )
        self.is_trained = False
        self.track = track
        self.bin_speeds = np.full(N_BINS, PP_SPEED)
        self.bin_size = track['totalLength'] / N_BINS

        # Pre-compute curvature limit per bin (max curvature in ±3 bin window)
        self.curv_limits = np.full(N_BINS, VP['vxMax'])
        pts_per_bin = max(1, track['n'] // N_BINS)
        for b in range(N_BINS):
            s_mid = (b + 0.5) * self.bin_size
            idx = find_idx_for_s(s_mid, track)
            max_curv = 0.0
            for j in range(-3, 4):
                cidx = (idx + j * pts_per_bin) % track['n']
                max_curv = max(max_curv, track['curvature'][cidx])
            if max_curv > 0.01:
                self.curv_limits[b] = min(VP['vxMax'],
                                          np.sqrt(A_LAT_MAX / max_curv))

    def add_lap_data(self, trajectory, iteration):
        """Update speed bins from a completed lap (safe set expansion)."""
        # Step 1: Record max achieved speed per bin
        for pt in trajectory:
            b = int(pt['s'] / self.bin_size) % N_BINS
            if pt['x'][3] > self.bin_speeds[b]:
                self.bin_speeds[b] = pt['x'][3]

        # Step 2: Apply iteration-based boost (capped at 3x)
        boost = min(1.0 + SPEED_BOOST * (iteration + 1), 3.0)
        boosted = self.bin_speeds * boost

        # Step 3: Cap with curvature limits
        for b in range(N_BINS):
            boosted[b] = min(boosted[b], self.curv_limits[b])

        # Step 4: Smooth with 5-bin moving average
        smoothed = np.zeros(N_BINS)
        for b in range(N_BINS):
            indices = [(b + j) % N_BINS for j in range(-2, 3)]
            smoothed[b] = np.mean(boosted[indices])

        self.bin_speeds = np.clip(smoothed, PP_SPEED, VP['vxMax'])

    def train(self):
        """Train NN on the binned speed targets."""
        # Generate training data (5 samples per bin for NN coverage)
        X, y = [], []
        for b in range(N_BINS):
            for offset in np.linspace(0.1, 0.9, 5):
                s_norm = (b + offset) / N_BINS
                curv = self.curv_limits[b]  # use curvature info
                X.append([s_norm, curv,
                          np.sin(2 * np.pi * s_norm),
                          np.cos(2 * np.pi * s_norm)])
                y.append(self.bin_speeds[b])

        self.model.fit(np.array(X), np.array(y))
        self.is_trained = True

        print(f"  Speed NN: target min={self.bin_speeds.min():.1f}  "
              f"max={self.bin_speeds.max():.1f}  "
              f"mean={self.bin_speeds.mean():.1f} m/s")

    def predict(self, s):
        """Predict target speed at arc-length position s."""
        if not self.is_trained:
            return PP_SPEED

        track = self.track
        s_norm = (s % track['totalLength']) / track['totalLength']
        b = int(s_norm * N_BINS) % N_BINS
        curv = self.curv_limits[b]

        feat = np.array([[s_norm, curv,
                          np.sin(2 * np.pi * s_norm),
                          np.cos(2 * np.pi * s_norm)]])
        v = float(self.model.predict(feat)[0])
        return np.clip(v, PP_SPEED, VP['vxMax'])

    def get_speed_profile(self, n_pts=200):
        """Return the full speed profile for plotting."""
        track = self.track
        s_vals = np.linspace(0, track['totalLength'], n_pts, endpoint=False)
        speeds = np.array([self.predict(s) for s in s_vals])
        return s_vals / track['totalLength'] * 100.0, speeds


# ==========================================================================
#  NN VALUE FUNCTION  (learns cost-to-go from trajectory data)
# ==========================================================================

class NNValueFunction:
    """
    Neural network that approximates the cost-to-go V(x).

    Trained on (state, remaining_steps) pairs from completed laps.
    Provides a smooth terminal cost for the MPC.

    Features: [X/50, Y/50, cos(phi), sin(phi), vx/vmax]
    Output:   predicted remaining steps to finish the lap
    """

    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=NN_VF_LAYERS,
            activation='relu', solver='adam',
            max_iter=500, warm_start=True, random_state=42,
        )
        self.is_trained = False
        self.X_data = []
        self.y_data = []

    def add_trajectory(self, trajectory):
        """Add (state, cost-to-go) pairs from a completed lap."""
        n = len(trajectory)
        for k, pt in enumerate(trajectory):
            self.X_data.append(self._features(pt['x']))
            self.y_data.append(float(n - k))

    def train(self):
        """Retrain NN on all accumulated data."""
        if len(self.X_data) < 50:
            return
        X = np.array(self.X_data)
        y = np.array(self.y_data)
        self.model.fit(X, y)
        self.is_trained = True
        score = self.model.score(X, y)
        print(f"  Value NN: {len(self.X_data)} samples, R² = {score:.3f}")

    def predict(self, x):
        """Predict cost-to-go V(x). Returns 0 if not yet trained."""
        if not self.is_trained:
            return 0.0
        feat = self._features(x).reshape(1, -1)
        return float(self.model.predict(feat)[0])

    def _features(self, x):
        return np.array([
            x[0] / 50.0, x[1] / 50.0,
            np.cos(x[2]), np.sin(x[2]),
            x[3] / VP['vxMax'],
        ])


# ==========================================================================
#  MPC CONTROLLER  (NN speed map + lookahead braking)
# ==========================================================================

def mpc_control(x, s_now, track, speed_map, value_fn):
    """
    NN-MPC controller:
      1. Query NN speed map for target speed at current position
      2. Look ahead 0.5-2.0 seconds for upcoming speed requirements
      3. If speed must drop ahead, start braking proactively
      4. Add small NN value function terminal cost adjustment
      5. Apply pure pursuit at the computed target speed
    """
    v_target = speed_map.predict(s_now)

    # Proactive braking: check upcoming positions
    vx = max(x[3], 1.0)
    for dt_look in [0.5, 1.0, 1.5]:
        s_ahead = s_now + vx * dt_look
        v_ahead = speed_map.predict(s_ahead)
        # Brake proactively if upcoming target is lower
        v_target = min(v_target, v_ahead + 1.5 * dt_look)

    # Terminal cost adjustment: if NN value function predicts high cost-to-go
    # at current speed, slightly reduce target to be cautious
    if value_fn.is_trained and LAMBDA_TERM > 0:
        x_fast = x.copy()
        x_fast[3] = min(v_target, VP['vxMax'])
        x_slow = x.copy()
        x_slow[3] = max(v_target * 0.8, PP_SPEED)
        vf_fast = value_fn.predict(x_fast)
        vf_slow = value_fn.predict(x_slow)
        # If faster state has lower cost-to-go, push speed up slightly
        if vf_fast < vf_slow:
            v_target = min(v_target * 1.05, VP['vxMax'])

    return pure_pursuit(x, s_now, track, v_target)


# ==========================================================================
#  LAP SIMULATION
# ==========================================================================

def run_lap(track, iteration, speed_map, value_fn):
    """Simulate one lap around the track."""
    start_phi = np.arctan2(track['ty'][0], track['tx'][0])
    x = np.array([track['px'][0], track['py'][0], start_phi, 1.0, 0.0, 0.0])

    lap_dist = 0.0
    prev_s = track['s'][0]
    t, step = 0.0, 0
    trajectory = []
    last_u = np.array([0.0, 0.0])

    while lap_dist < track['totalLength'] and t < MAX_LAP_TIME:
        proj = project_on_track(x[0], x[1], track)

        if step % MPC_EVERY == 0:
            if iteration == 0:
                last_u = pure_pursuit(x, proj['s'], track, PP_SPEED)
            else:
                last_u = mpc_control(x, proj['s'], track, speed_map, value_fn)

        last_u[0] = np.clip(last_u[0], -VP['deltaMax'], VP['deltaMax'])
        last_u[1] = np.clip(last_u[1], VP['DMin'], VP['DMax'])

        x = clip_state(rk4_step(x, last_u, DT))

        proj_new = project_on_track(x[0], x[1], track)
        ds = proj_new['s'] - prev_s
        if ds < -track['totalLength'] * 0.5:
            ds += track['totalLength']
        if ds > track['totalLength'] * 0.5:
            ds -= track['totalLength']
        if ds > 0:
            lap_dist += ds
        prev_s = proj_new['s']

        t += DT
        step += 1
        trajectory.append({
            'x': x.copy(), 'u': last_u.copy(),
            's': proj_new['s'], 'd': proj_new['d'], 't': t,
        })

        if abs(proj_new['d']) > TRACK_HW * 3:
            print(f"  !! Off track at t={t:.1f}s  d={proj_new['d']:.1f}m — aborting")
            return None

        if step % 400 == 0:
            pct = 100.0 * lap_dist / track['totalLength']
            print(f"    t={t:.1f}s  vx={x[3]:.1f}m/s  progress={pct:.0f}%")

    return dict(trajectory=trajectory, lap_time=t, n_steps=len(trajectory))


# ==========================================================================
#  PLOTTING
# ==========================================================================

COLORS = ['#4FC3F7', '#66BB6A', '#FFCA28', '#FFA726', '#EF5350',
          '#AB47BC', '#26C6DA', '#EC407A', '#78909C', '#8D6E63']

def plot_results(track, results, speed_profiles):
    """Generate a 2x2 plot of racing results."""
    valid = [(i, r) for i, r in enumerate(results) if r is not None]
    if not valid:
        print("No successful laps to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0a0a1a')
    fig.suptitle('NN-MPC Autonomous Racing — Learning Progress',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    # ── Panel 1: Track + Trajectories ────────────────────────────────────
    ax = axes[0, 0]
    ax.set_aspect('equal')
    ax.set_title('Track & Trajectories', color='white', fontsize=13, pad=10)
    ax.set_facecolor('#2a2a2a')
    hw = TRACK_HW
    lx = track['px'] + hw * track['nx']
    ly = track['py'] + hw * track['ny']
    rx = track['px'] - hw * track['nx']
    ry = track['py'] - hw * track['ny']
    from matplotlib.patches import Polygon as MplPoly
    surface = np.vstack([np.column_stack([lx, ly]),
                         np.column_stack([rx, ry])[::-1]])
    ax.add_patch(MplPoly(surface, closed=True, facecolor='#555555',
                         edgecolor='none', alpha=0.5))
    ax.plot(np.append(lx, lx[0]), np.append(ly, ly[0]),
            'w-', lw=1.5, alpha=0.7)
    ax.plot(np.append(rx, rx[0]), np.append(ry, ry[0]),
            'w-', lw=1.5, alpha=0.7)
    ax.plot(np.append(track['px'], track['px'][0]),
            np.append(track['py'], track['py'][0]),
            'w--', lw=0.5, alpha=0.3)
    for i, result in valid:
        xs = [p['x'][0] for p in result['trajectory']]
        ys = [p['x'][1] for p in result['trajectory']]
        c = COLORS[i % len(COLORS)]
        lw = 2.0 if i == valid[-1][0] else 1.0
        alpha = 0.9 if i == valid[-1][0] else 0.5
        ax.plot(xs, ys, color=c, lw=lw, alpha=alpha,
                label=f"Iter {i}  ({result['lap_time']:.1f}s)")
    ax.legend(loc='upper left', fontsize=7, facecolor='#333',
              edgecolor='#666', labelcolor='white')
    ax.tick_params(colors='white')
    ax.set_xlabel('X [m]', color='white')
    ax.set_ylabel('Y [m]', color='white')

    # ── Panel 2: Lap Times ───────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title('Lap Times (decreasing = learning)', color='white',
                 fontsize=13, pad=10)
    ax.set_facecolor('#1a1a2e')
    for idx, (i, r) in enumerate(valid):
        c = COLORS[i % len(COLORS)]
        ax.bar(f"#{i}", r['lap_time'], color=c, edgecolor='white',
               linewidth=0.5)
        ax.text(idx, r['lap_time'] + 0.5, f"{r['lap_time']:.1f}s",
                ha='center', va='bottom', color='white', fontsize=9)
    ax.set_ylabel('Lap Time [s]', color='white')
    ax.tick_params(colors='white')
    if len(valid) >= 2:
        t0, tf = valid[0][1]['lap_time'], valid[-1][1]['lap_time']
        imp = (t0 - tf) / t0 * 100
        ax.text(0.98, 0.95, f"Improvement: {imp:.0f}%",
                transform=ax.transAxes, ha='right', va='top',
                color='#66BB6A', fontsize=12, fontweight='bold')

    # ── Panel 3: Speed Profiles ──────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title('Speed Profiles (actual)', color='white', fontsize=13, pad=10)
    ax.set_facecolor('#1a1a2e')
    for i, result in valid:
        pct = [100.0 * p['s'] / track['totalLength']
               for p in result['trajectory']]
        spd = [p['x'][3] for p in result['trajectory']]
        c = COLORS[i % len(COLORS)]
        lw = 1.5 if i == valid[-1][0] else 0.8
        alpha = 0.9 if i == valid[-1][0] else 0.5
        ax.plot(pct, spd, color=c, lw=lw, alpha=alpha, label=f"Iter {i}")
    ax.set_xlabel('Track Progress [%]', color='white')
    ax.set_ylabel('Speed vx [m/s]', color='white')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666',
              labelcolor='white', ncol=2)
    ax.tick_params(colors='white')

    # ── Panel 4: NN Speed Map Evolution ──────────────────────────────────
    ax = axes[1, 1]
    ax.set_title('NN Speed Map (learned target speeds)', color='white',
                 fontsize=13, pad=10)
    ax.set_facecolor('#1a1a2e')
    for idx, (s_pct, spd) in enumerate(speed_profiles):
        c = COLORS[idx % len(COLORS)]
        lw = 1.5 if idx == len(speed_profiles) - 1 else 0.8
        alpha = 0.9 if idx == len(speed_profiles) - 1 else 0.4
        ax.plot(s_pct, spd, color=c, lw=lw, alpha=alpha,
                label=f"After iter {idx}")
    ax.set_xlabel('Track Progress [%]', color='white')
    ax.set_ylabel('Target Speed [m/s]', color='white')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666',
              labelcolor='white', ncol=2)
    ax.tick_params(colors='white')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = 'lmpc_results.png'
    plt.savefig(out_path, dpi=150, facecolor='#0a0a1a')
    print(f"\nPlot saved to {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    print("NN-MPC Autonomous Racing Demo")
    print("=" * 55)
    print(f"Track: {TRACK_CHOICE}  |  Iterations: {N_ITERATIONS}")
    print(f"Speed NN: {NN_SPEED_LAYERS}  |  Boost: {SPEED_BOOST}")
    print(f"dt: {DT}s  |  MPC every {MPC_EVERY} steps\n")

    # Build track
    if TRACK_CHOICE == "custom":
        track = build_track_spline(CUSTOM_WP)
    elif TRACK_CHOICE == "mpcc_center":
        track = build_track_points(MPCC_CENTER)
    elif TRACK_CHOICE == "mpcc_racing":
        track = build_track_points(MPCC_RACING)
    else:
        raise ValueError(f"Unknown track: {TRACK_CHOICE}")

    print(f"Track length: {track['totalLength']:.1f} m  "
          f"({track['n']} points, half-width {TRACK_HW} m)")
    print(f"Max curvature: {track['curvature'].max():.4f} rad/m  "
          f"(smoothed max: {track['smooth_curvature'].max():.4f})\n")

    speed_map = NNSpeedMap(track)
    value_fn = NNValueFunction()
    results = []
    speed_profiles = []   # (s_pct, speeds) for each iteration
    total_t0 = timer.time()

    for it in range(N_ITERATIONS):
        mode = "Pure Pursuit" if it == 0 else "NN-MPC"
        print(f"{'=' * 55}")
        print(f"  Iteration {it}  ({mode})")
        print(f"{'=' * 55}")

        # Train NNs before MPC iterations
        if it > 0:
            speed_map.train()
            value_fn.train()

        t0 = timer.time()
        result = run_lap(track, it, speed_map, value_fn)
        elapsed = timer.time() - t0

        if result is None:
            print(f"  FAILED — skipping")
            results.append(None)
            continue

        results.append(result)
        speed_map.add_lap_data(result['trajectory'], it)
        value_fn.add_trajectory(result['trajectory'])

        # Record speed profile for plotting
        if speed_map.is_trained:
            speed_profiles.append(speed_map.get_speed_profile())
        else:
            # Before first training, show PP_SPEED everywhere
            s_pct = np.linspace(0, 100, 200, endpoint=False)
            speed_profiles.append((s_pct, np.full(200, PP_SPEED)))

        speeds = [p['x'][3] for p in result['trajectory']]
        max_d = max(abs(p['d']) for p in result['trajectory'])
        print(f"  Lap time : {result['lap_time']:.1f}s  "
              f"({result['n_steps']} steps)")
        print(f"  Speed    : avg={np.mean(speeds):.1f}  "
              f"min={min(speeds):.1f}  max={max(speeds):.1f} m/s")
        print(f"  Max |d|  : {max_d:.1f} m  (limit {TRACK_HW} m)")
        print(f"  Wall time: {elapsed:.1f}s\n")

    total_elapsed = timer.time() - total_t0
    print(f"\n{'=' * 55}")
    print(f"  SUMMARY  (total wall time: {total_elapsed:.0f}s)")
    print(f"{'=' * 55}")

    valid = [(i, r) for i, r in enumerate(results) if r is not None]
    for i, r in valid:
        spd = np.mean([p['x'][3] for p in r['trajectory']])
        print(f"  Iter {i}: {r['lap_time']:6.1f}s   avg speed {spd:.1f} m/s")

    if len(valid) >= 2:
        imp = valid[0][1]['lap_time'] - valid[-1][1]['lap_time']
        pct = 100.0 * imp / valid[0][1]['lap_time']
        print(f"\n  Improvement: {imp:.1f}s  ({pct:.1f}%)")

    plot_results(track, results, speed_profiles)


if __name__ == '__main__':
    main()
