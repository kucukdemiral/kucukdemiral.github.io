%% Koopman Cart-Pole Hybrid Demo
% Standalone MATLAB script for the learned cart-pole example.
%
% Method:
%   1. Generate broad EDMD data from the nonlinear cart-pole simulator.
%   2. Learn a dedicated local balance model near upright.
%   3. Use energy-based swing-up away from upright.
%   4. Use Koopman MPC only inside a tight local balance region.
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
%% Learn broad EDMD model for consistency with the notes
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

%% Learn local balance model used by the actual MPC controller
[A_bal, B_bal] = train_local_balance_model(Ts, Tc, mc, mp, lp, g);
Q_bal = diag([15, 2, 120, 12]);
R_bal = 0.3;
[P_bal, ~, ~] = dare(A_bal, B_bal, Q_bal, R_bal);
N_bal = 30;

%% Closed-loop simulation
s = [x0; xd0; deg2rad(theta0_deg); thd0];
N_sim = round(T_final / Tc);
states = zeros(4, N_sim + 1);
controls = zeros(1, N_sim);
times = (0:N_sim) * Tc;
states(:, 1) = s;

for k = 1:N_sim
    th = wrap_angle(s(3));
    if abs(th) < 0.035 && abs(s(4)) < 0.22
        u = local_balance_mpc_control(s, A_bal, B_bal, Q_bal, R_bal, P_bal, N_bal, u_max);
    else
        inertia = mp * lp^2;
        energy = 0.5 * inertia * s(4)^2 + mp * g * lp * (cos(s(3)) - 1);
        if abs(th) < 0.25
            swing_gain = 15;
        else
            swing_gain = 35;
        end
        swing = swing_gain * energy * sign(s(4) * cos(s(3)) + 1e-4);
        centre = -1.0 * s(1) - 2.0 * s(2) - 2.0 * sin(s(3)) - 1.0 * s(4) * cos(s(3));
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

function [A_bal, B_bal] = train_local_balance_model(Ts, Tc, mc, mp, lp, g)
    rng(12362);
    Zx = zeros(12000, 5);
    Zy = zeros(12000, 4);
    for sample = 1:12000
        s = [0.15 * (rand - 0.5) * 2;
             0.30 * (rand - 0.5) * 2;
             0.03 * (rand - 0.5) * 2;
             0.25 * (rand - 0.5) * 2];
        u = 2.0 * (rand - 0.5);
        z = [s(1); s(2); wrap_angle(s(3)); s(4)];
        sn = sim_step(s, u, Ts, Tc, mc, mp, lp, g);
        zn = [sn(1); sn(2); wrap_angle(sn(3)); sn(4)];
        Zx(sample, :) = [z.', u];
        Zy(sample, :) = zn.';
    end

    W = (Zx' * Zx + 1e-10 * eye(5)) \ (Zx' * Zy);
    A_bal = W(1:4, :).';
    B_bal = W(5, :).';
end

function u = local_balance_mpc_control(s, A_bal, B_bal, Q_bal, R_bal, P_bal, N, u_max)
    x0 = [s(1); s(2); wrap_angle(s(3)); s(4)];

    Apow = cell(N + 1, 1);
    Apow{1} = eye(4);
    for k = 2:N+1
        Apow{k} = A_bal * Apow{k - 1};
    end

    GB = cell(N + 1, 1);
    for k = 1:N+1
        GB{k} = Apow{k} * B_bal;
    end

    free = cell(N + 1, 1);
    free{1} = x0;
    for k = 2:N+1
        free{k} = A_bal * free{k - 1};
    end

    H = zeros(N, N);
    f = zeros(N, 1);
    for i = 1:N
        for j = i:N
            hs = 0;
            for k = j+1:N+1
                if k <= N
                    Wk = Q_bal;
                else
                    Wk = P_bal;
                end
                hs = hs + GB{k - i}' * Wk * GB{k - j};
            end
            if i == j
                hs = hs + R_bal;
            end
            H(i, j) = hs;
            H(j, i) = hs;
        end

        fs = 0;
        for k = i+1:N+1
            if k <= N
                Wk = Q_bal;
            else
                Wk = P_bal;
            end
            fs = fs + GB{k - i}' * Wk * free{k};
        end
        f(i) = fs;
    end

    U_free = -H \ f;
    if all(abs(U_free) <= u_max + 1e-9)
        u = U_free(1);
        return;
    end

    U = min(max(U_free, -u_max), u_max);
    alpha = 1 / (max(sum(abs(H), 2)) + 1e-8);
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
