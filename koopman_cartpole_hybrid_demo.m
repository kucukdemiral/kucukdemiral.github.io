%% Koopman Cart-Pole Hybrid Demo
% Standalone MATLAB script for the learned cart-pole example.
%
% Method:
%   1. Generate data from the nonlinear cart-pole simulator.
%   2. Learn a Koopman predictor with EDMD.
%   3. Use energy-based swing-up away from upright.
%   4. Use local Koopman MPC near upright, blended with a local stabilizer.
%
% Edit the USER SETTINGS section below and run the file.

clear; clc; close all; rng(12345);

%% USER SETTINGS
theta0_deg = 180;
x0 = 0.0;
xd0 = 0.0;
thd0 = 0.0;

mc = 1.0;
mp = 0.3;
lp = 0.5;
g = 9.81;
Ts = 0.02;
Tc = 0.04;
T_final = 10.0;
u_max = 15.0;

n_traj = 200;
traj_len = 25;
n_mpc = 16;

%% Learn Koopman predictor with EDMD
p_lift = 10;
Zx = zeros(n_traj * traj_len, p_lift + 1);
Zy = zeros(n_traj * traj_len, p_lift);
row = 1;

for traj = 1:n_traj
    s = [2*(rand-0.5);
         2*(rand-0.5);
         2*pi*(rand-0.5);
         4*(rand-0.5)];
    for k = 1:traj_len
        u = 2*u_max*(rand-0.5);
        z_now = lift_state(s);
        s_next = sim_step(s, u, Ts, Tc, mc, mp, lp, g);
        z_next = lift_state(s_next);
        Zx(row, :) = [z_now.', u];
        Zy(row, :) = z_next.';
        s = s_next;
        row = row + 1;
    end
end

XtX = Zx' * Zx + 1e-6 * eye(p_lift + 1);
XtY = Zx' * Zy;
W = XtX \ XtY;
A_lift = W(1:p_lift, :).';
B_lift = W(end, :).';

%% Local stabilizer near upright
[A_lin, B_lin] = linearize_cartpole_step(Ts, Tc, mc, mp, lp, g);
Q_local = diag([15, 2, 120, 12]);
R_local = 0.3;
[P_local, ~, ~] = dare(A_lin, B_lin, Q_local, R_local);
K_local = (R_local + B_lin' * P_local * B_lin) \ (B_lin' * P_local * A_lin);

z_ref = lift_state([0; 0; 0; 0]);
lift_offset = A_lift * z_ref - z_ref;
Q_diag = [18; 2; 22; 3; 18; 10; 0; 0; 0; 0];
Qf_diag = 6 * Q_diag;

%% Closed-loop simulation
s = [x0; xd0; deg2rad(theta0_deg); thd0];
N_sim = round(T_final / Tc);
states = zeros(4, N_sim + 1);
controls = zeros(1, N_sim);
times = (0:N_sim) * Tc;
states(:, 1) = s;

for k = 1:N_sim
    th = wrap_angle(s(3));
    if abs(th) < 0.35 && abs(s(4)) < 2.0
        u_mpc = koopman_mpc_control(s, A_lift, B_lift, z_ref, lift_offset, n_mpc, Q_diag, 0.2, Qf_diag, u_max);
        e = [s(1); s(2); wrap_angle(s(3)); s(4)];
        u_local = max(-u_max, min(u_max, -K_local * e));
        u = max(-u_max, min(u_max, 0.15 * u_mpc + 0.85 * u_local));
    else
        inertia = mp * lp^2;
        energy = 0.5 * inertia * s(4)^2 + mp * g * lp * (cos(s(3)) - 1);
        swing = 35 * energy * sign(s(4) * cos(s(3)) + 1e-4);
        centre = -1.0 * s(1) - 2.0 * s(2);
        u = max(-u_max, min(u_max, swing + centre));
    end

    controls(k) = u;
    s = sim_step(s, u, Ts, Tc, mc, mp, lp, g);
    states(:, k + 1) = s;
end

%% Plot results
theta_deg = arrayfun(@(a) rad2deg(wrap_angle(a)), states(3, :));

figure('Color', 'w', 'Position', [100 100 900 850]);
tiledlayout(4, 1, 'TileSpacing', 'compact');

nexttile;
plot(times, states(1, :), 'LineWidth', 1.8, 'Color', [0.90 0.30 0.20]); hold on;
yline(0, '--', 'Color', 0.7);
ylabel('x (m)'); grid on;

nexttile;
plot(times, states(2, :), 'LineWidth', 1.8, 'Color', [0.60 0.35 0.75]); hold on;
yline(0, '--', 'Color', 0.7);
ylabel('xdot (m/s)'); grid on;

nexttile;
plot(times, theta_deg, 'LineWidth', 1.8, 'Color', [0.20 0.50 0.90]); hold on;
yline(0, '--', 'Color', 0.7);
ylabel('theta (deg)'); grid on;

nexttile;
stairs(times(1:end-1), controls, 'LineWidth', 1.8, 'Color', [0.95 0.60 0.10]); hold on;
yline(u_max, '--', 'Color', 0.7);
yline(-u_max, '--', 'Color', 0.7);
ylabel('u (N)'); xlabel('Time (s)'); grid on;

sgtitle('Cart-Pole Hybrid Swing-Up with Koopman MPC');

fprintf('Final state = [%.4f %.4f %.4f %.4f]\\n', states(:, end));
fprintf('Max |x| = %.4f m\\n', max(abs(states(1, :))));

%% Local functions
function z = lift_state(s)
    x = s(1); xd = s(2); th = s(3); thd = s(4);
    sth = sin(th); cth = cos(th);
    z = [x; xd; th; thd; sth; cth; xd*sth; xd*cth; thd*sth; thd*cth];
end

function a = wrap_angle(a)
    while a > pi
        a = a - 2*pi;
    end
    while a < -pi
        a = a + 2*pi;
    end
end

function ds = cartpole_deriv(s, u, mc, mp, lp, g)
    x = s(1); xd = s(2); th = s(3); thd = s(4);
    sth = sin(th); cth = cos(th);
    mt = mc + mp;
    denom = mt - mp * cth^2;
    thdd = (g * sth * mt - cth * (u + mp * lp * thd^2 * sth)) / (lp * denom);
    xdd = (u + mp * lp * (thd^2 * sth - thdd * cth)) / mt;
    ds = [xd; xdd; thd; thdd];
end

function sn = rk4_step(s, u, dt, mc, mp, lp, g)
    k1 = cartpole_deriv(s, u, mc, mp, lp, g);
    k2 = cartpole_deriv(s + 0.5 * dt * k1, u, mc, mp, lp, g);
    k3 = cartpole_deriv(s + 0.5 * dt * k2, u, mc, mp, lp, g);
    k4 = cartpole_deriv(s + dt * k3, u, mc, mp, lp, g);
    sn = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
    sn(3) = wrap_angle(sn(3));
end

function sn = sim_step(s, u, Ts, Tc, mc, mp, lp, g)
    n_steps = round(Tc / Ts);
    sn = s;
    for ii = 1:n_steps
        sn = rk4_step(sn, u, Ts, mc, mp, lp, g);
    end
    sn(3) = wrap_angle(sn(3));
end

function [A, B] = linearize_cartpole_step(Ts, Tc, mc, mp, lp, g)
    x0 = zeros(4, 1);
    eps = 1e-6;
    A = zeros(4, 4);
    B = zeros(4, 1);
    for j = 1:4
        xp = x0; xm = x0;
        xp(j) = xp(j) + eps;
        xm(j) = xm(j) - eps;
        fp = sim_step(xp, 0, Ts, Tc, mc, mp, lp, g);
        fm = sim_step(xm, 0, Ts, Tc, mc, mp, lp, g);
        A(:, j) = (fp - fm) / (2 * eps);
    end
    fp = sim_step(x0, eps, Ts, Tc, mc, mp, lp, g);
    fm = sim_step(x0, -eps, Ts, Tc, mc, mp, lp, g);
    B = (fp - fm) / (2 * eps);
end

function u = koopman_mpc_control(s, A_lift, B_lift, z_ref, lift_offset, N, Q_diag, R_val, Qf_diag, u_max)
    p_lift = numel(z_ref);
    z = lift_state(s);
    e0 = z - z_ref;

    Apow = cell(N + 1, 1);
    Apow{1} = eye(p_lift);
    for k = 2:N+1
        Apow{k} = A_lift * Apow{k - 1};
    end

    GB = cell(N + 1, 1);
    for k = 1:N+1
        GB{k} = Apow{k} * B_lift;
    end

    free = cell(N + 1, 1);
    free{1} = e0;
    for k = 2:N+1
        free{k} = A_lift * free{k - 1} + lift_offset;
    end

    H = zeros(N, N);
    f = zeros(N, 1);
    for i = 1:N
        for j = i:N
            hs = 0;
            for k = j+1:N+1
                if k <= N
                    w = Q_diag;
                else
                    w = Qf_diag;
                end
                hs = hs + sum(GB{k - i} .* w .* GB{k - j});
            end
            if i == j
                hs = hs + R_val;
            end
            H(i, j) = hs;
            H(j, i) = hs;
        end

        fs = 0;
        for k = i+1:N+1
            if k <= N
                w = Q_diag;
            else
                w = Qf_diag;
            end
            fs = fs + sum(GB{k - i} .* w .* free{k});
        end
        f(i) = fs;
    end

    U = zeros(N, 1);
    alpha = 1 / (max(diag(H)) + 1e-8);
    for iter = 1:800
        grad = H * U + f;
        U_new = min(max(U - alpha * grad, -u_max), u_max);
        if max(abs(U_new - U)) < 1e-8
            U = U_new;
            break;
        end
        U = U_new;
    end
    u = U(1);
end
