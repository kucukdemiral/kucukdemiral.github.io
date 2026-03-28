%% Offset-Free MPC with Disturbance Estimation
%  Companion script for Chapter 8 (Output Feedback & Offset-Free MPC)
%  of the MPC lecture notes by I. Kucukdemiral.
%
%  Requirements: YALMIP (https://yalmip.github.io) + quadprog (Optimization Toolbox)
%  Install YALMIP in MATLAB Online via Add-Ons > Get Add-Ons > search "YALMIP"
%
%  This script demonstrates:
%    1. Augmented-state observer (Kalman filter with disturbance state)
%    2. Steady-state target calculator
%    3. Deviation-form MPC with soft output constraints
%    4. Closed-loop simulation with piecewise-constant reference & disturbance

clear; clc; close all; rng(1);

%% 1. System Definition
A  = [0.9535  0.0761; -0.8454  0.5478];
B  = [0.0465 ; 0.8454];
Bw = B;
C  = [1 0];

nx = 2; nu = 1; ny = 1;

u_min = -5;  u_max =  5;
y_min = -0.5; y_max =  0.5;

%% 2. Augmented Observer
% Augmented state: z_k = [x_k; d_k],  d_{k+1} = d_k
A_aug = [A, Bw; zeros(1,nx), 1];
B_aug = [B; 0];
C_aug = [C, 0];

% W(3,3) is the single tuning knob for disturbance tracking bandwidth
W = diag([1e-4, 1e-4, 1e-4]);   % Kalman filter tuning weights
V = 1e-3;                        % Measurement noise variance

[P_ss, ~, ~] = dare(A_aug', C_aug', W, V);
L = P_ss * C_aug' / (C_aug * P_ss * C_aug' + V);

fprintf('Observer gain L = [%.4f; %.4f; %.4f]\n', L);

%% 3. Target Calculator
% [I-A  -B] [x_ss]   [Bw * d_hat]
% [ C    0] [u_ss] = [   ref    ]
M_target = [eye(nx)-A, -B; C, zeros(ny,nu)];

if rank(M_target) < (nx + nu)
    error('Target calculator matrix is rank deficient.');
end
fprintf('Target calculator: rank = %d (full rank)\n', rank(M_target));

%% 4. MPC Formulation
N   = 20;
q_y = 10;
R   = 0.1;
rho = 1e5;

Q_state = C' * q_y * C;
[P_N, ~, ~] = dare(A, B, Q_state, R);

du    = sdpvar(nu, N,   'full');
dx    = sdpvar(nx, N+1, 'full');
slack = sdpvar(ny, N,   'full');

x_hat = sdpvar(nx, 1);
x_ss  = sdpvar(nx, 1);
u_ss  = sdpvar(nu, 1);

obj  = 0;
cons = [dx(:,1) == x_hat - x_ss];

for k = 1:N
    cons = [cons, dx(:,k+1) == A*dx(:,k) + B*du(:,k)];

    dy_k = C * dx(:,k);
    obj  = obj + dy_k'*q_y*dy_k + du(:,k)'*R*du(:,k) + rho*slack(:,k)'*slack(:,k);

    u_abs = du(:,k) + u_ss;
    y_abs = C * (dx(:,k) + x_ss);

    cons = [cons, u_min <= u_abs <= u_max];
    cons = [cons, y_min - slack(:,k) <= y_abs <= y_max + slack(:,k)];
    cons = [cons, slack(:,k) >= 0];
end

obj = obj + dx(:,N+1)' * P_N * dx(:,N+1);

ops = sdpsettings('solver', 'quadprog', 'verbose', 0);
mpc_controller = optimizer(cons, obj, ops, {x_hat, x_ss, u_ss}, {du(:,1), slack(:,1)});
fprintf('MPC controller compiled (N = %d, solver = quadprog)\n', N);

%% 5. Simulation Setup
T_sim = 300;
x_true = zeros(nx, 1);
z_hat  = zeros(nx+1, 1);
u_applied = 0;

ref_signal = [repmat( 0.4,1,100), repmat(-0.4,1,100), repmat( 0.4,1,100)];
w_true     = [repmat( 0.2,1,150), repmat(-0.2,1,150)];

log_y_true = zeros(1,T_sim);  log_y_meas = zeros(1,T_sim);
log_ref    = zeros(1,T_sim);  log_u      = zeros(1,T_sim);
log_u_ss   = zeros(1,T_sim);  log_d_true = zeros(1,T_sim);
log_d_hat  = zeros(1,T_sim);  log_slack  = zeros(1,T_sim);
log_x1     = zeros(1,T_sim);  log_x2     = zeros(1,T_sim);

fprintf('Starting simulation (%d steps)...\n', T_sim);

%% 6. Simulation Loop
for t = 1:T_sim

    ref = ref_signal(t);

    % A. Plant output
    y_true = C * x_true;
    y_meas = y_true + sqrt(V) * randn;

    % B. Observer
    z_hat_pred = A_aug * z_hat + B_aug * u_applied;
    z_hat      = z_hat_pred + L * (y_meas - C_aug * z_hat_pred);

    x_hat_k = z_hat(1:nx);
    d_hat_k = z_hat(nx+1);

    % C. Target calculator
    targets = M_target \ [Bw * d_hat_k; ref];
    x_ss_k  = targets(1:nx);
    u_ss_k  = targets(nx+1:end);

    % D. MPC
    [sol, diagnostics] = mpc_controller{x_hat_k, x_ss_k, u_ss_k};

    if diagnostics == 0
        du_opt    = full(sol{1});
        slack_now = full(sol{2});
    else
        warning('MPC infeasible at t = %d (code %d). Applying u_ss.', t, diagnostics);
        du_opt    = 0;
        slack_now = NaN;
    end

    u_applied = min(max(du_opt + u_ss_k, u_min), u_max);

    % E. Plant update
    x_true = A * x_true + B * u_applied + Bw * w_true(t);

    % F. Logging
    log_y_true(t) = y_true;  log_y_meas(t) = y_meas;
    log_ref(t)    = ref;     log_u(t)      = u_applied;
    log_u_ss(t)   = u_ss_k;  log_d_true(t) = w_true(t);
    log_d_hat(t)  = d_hat_k; log_slack(t)  = slack_now;
    log_x1(t)     = x_true(1); log_x2(t)   = x_true(2);
end

fprintf('Simulation complete.\n');

%% 7. Visualisation
t_vec = 1:T_sim;

figure('Position', [100 100 900 700], 'Color', 'w');

% Panel 1: Output tracking
subplot(4,1,1);
plot(t_vec, log_y_true, 'b', 'LineWidth', 1.2); hold on;
plot(t_vec, log_y_meas, 'c.', 'MarkerSize', 2);
stairs(t_vec, log_ref, 'r--', 'LineWidth', 1.2);
yline(y_max, 'k--', 'LineWidth', 0.8);
yline(y_min, 'k--', 'LineWidth', 0.8);
ylabel('y'); title('Offset-Free MPC: Output Tracking');
legend('y_{true}', 'y_{meas}', 'r_{ref}', 'y bounds', 'Location', 'best');
grid on; xlim([1 T_sim]);

% Panel 2: Control input
subplot(4,1,2);
stairs(t_vec, log_u, 'b', 'LineWidth', 1.2); hold on;
stairs(t_vec, log_u_ss, 'r--', 'LineWidth', 1);
yline(u_max, 'k--', 'LineWidth', 0.8);
yline(u_min, 'k--', 'LineWidth', 0.8);
ylabel('u'); legend('u_{applied}', 'u_{ss}', 'Location', 'best');
grid on; xlim([1 T_sim]);

% Panel 3: Disturbance estimation
subplot(4,1,3);
stairs(t_vec, log_d_true, 'r', 'LineWidth', 1.2); hold on;
plot(t_vec, log_d_hat, 'b', 'LineWidth', 1.2);
ylabel('d'); legend('d_{true}', '\hat{d}', 'Location', 'best');
grid on; xlim([1 T_sim]);

% Panel 4: Soft constraint slack
subplot(4,1,4);
stem(t_vec, log_slack, 'Marker', 'none', 'Color', [0.8 0.2 0.2], 'LineWidth', 0.8);
ylabel('\epsilon'); xlabel('Time step k');
title('Soft Output Constraint Activity');
grid on; xlim([1 T_sim]);

sgtitle('Offset-Free MPC with Augmented Disturbance Observer', 'FontSize', 14, 'FontWeight', 'bold');
