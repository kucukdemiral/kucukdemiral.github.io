%% Offset-Free MPC with Disturbance Estimation
%  Companion script for Chapter 8 (Output Feedback & Offset-Free MPC)
%  of the MPC lecture notes by I. Kucukdemiral.
%
%  Requirements: Optimization Toolbox (quadprog) — included in MATLAB Online.
%  No additional toolboxes or Add-Ons needed.
%
%  This script demonstrates:
%    1. Augmented-state observer (Kalman filter with disturbance state)
%    2. Steady-state target calculator
%    3. Deviation-form MPC with soft output constraints (QP via quadprog)
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
Wn = diag([1e-4, 1e-4, 1e-4]);  % Kalman filter process noise
V  = 1e-3;                       % Measurement noise variance

[P_ss, ~, ~] = dare(A_aug', C_aug', Wn, V);
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

%% 4. MPC QP Formulation (direct quadprog — no YALMIP needed)
N   = 20;          % prediction horizon
q_y = 10;          % output tracking weight
R   = 0.1;         % input weight
rho = 1e5;         % soft constraint penalty

Q_state = C' * q_y * C;               % nx x nx state weight
[P_N, ~, ~] = dare(A, B, Q_state, R); % terminal cost from DARE

% Decision variable: z = [du(1); ...; du(N); eps(1); ...; eps(N)]
%   du(k) in R^nu, eps(k) in R^ny
nz = N * (nu + ny);

% --- Build prediction matrices: dx(k) = Phi(k)*dx0 + Gamma(k)*[du(1);...;du(k-1)]
% where dx0 = x_hat - x_ss (the initial deviation)
% We need dx(k) for k = 1,...,N+1 and du for k = 1,...,N
% dx(1) = dx0,  dx(k+1) = A*dx(k) + B*du(k)

% Propagation: dx(k) = A^(k-1)*dx0 + sum_{j=1}^{k-1} A^(k-1-j)*B*du(j)
% Build Phi (state from initial condition) and Gamma (state from inputs)
Phi   = zeros(nx*(N+1), nx);    % maps dx0 -> [dx(1);...;dx(N+1)]
Gamma = zeros(nx*(N+1), nu*N);  % maps [du(1);...;du(N)] -> [dx(1);...;dx(N+1)]

A_pow = eye(nx);
for k = 1:N+1
    rows = (k-1)*nx+1 : k*nx;
    Phi(rows, :) = A_pow;
    for j = 1:min(k-1, N)
        cols = (j-1)*nu+1 : j*nu;
        Gamma(rows, cols) = A_pow / (A^j) * (A^(j-1)) * B;
    end
    if k <= N
        A_pow = A_pow * A;
    end
end

% Recompute Gamma more carefully using the standard approach
Gamma = zeros(nx*(N+1), nu*N);
for k = 2:N+1    % dx(k) depends on du(1),...,du(k-1)
    rows = (k-1)*nx+1 : k*nx;
    for j = 1:k-1
        cols = (j-1)*nu+1 : j*nu;
        Gamma(rows, cols) = A^(k-1-j) * B;
    end
end

% --- Hessian H and linear term f(dx0) ---
% Cost = sum_{k=1}^{N} [dx(k)'*Q*dx(k) + du(k)'*R*du(k) + rho*eps(k)'*eps(k)]
%      + dx(N+1)' * P_N * dx(N+1)
% With dx depending linearly on dx0 and du, we can write:
%   Cost = z' * H * z + f' * z + const
% where z = [du(1);...;du(N); eps(1);...;eps(N)]

% Build block-diagonal weight matrices for states
Q_blk = blkdiag(kron(eye(N), Q_state), P_N);  % (N+1)*nx x (N+1)*nx

% Hessian contributions from states: Gamma' * Q_blk * Gamma
% Only du part of z contributes to state cost via Gamma
H_du_du = Gamma' * Q_blk * Gamma + kron(eye(N), R);  % N*nu x N*nu
H_eps   = rho * eye(N*ny);                            % N*ny x N*ny

H = blkdiag(H_du_du, H_eps);
H = (H + H') / 2;  % ensure symmetry

% Linear term: f depends on dx0 (computed at each time step)
% f_du = 2 * Gamma' * Q_blk * Phi * dx0   (but quadprog uses 0.5*z'*H*z + f'*z)
% So f_du(dx0) = Gamma' * Q_blk * Phi * dx0  (the factor of 2 is in quadprog)

% We precompute the constant part:
f_du_matrix = Gamma' * Q_blk * Phi;  % N*nu x nx
% f_eps is zero (no linear term for slack)

% --- Inequality constraints: Ain * z <= bin(dx0, x_ss, u_ss) ---
% Constraints:
%   (a) u_min <= du(k) + u_ss <= u_max   =>  du(k) <= u_max - u_ss
%                                             -du(k) <= -u_min + u_ss
%   (b) y_min - eps(k) <= C*dx(k) + C*x_ss <= y_max + eps(k)
%       => C*dx(k) - eps(k) <= y_max - C*x_ss
%       => -C*dx(k) - eps(k) <= -y_min + C*x_ss
%   (c) -eps(k) <= 0

% Extract submatrices for dx(1),...,dx(N) from Phi and Gamma
% (not dx(N+1) — output constraints apply at prediction steps 1..N)
Phi_pred   = Phi(1:nx*N, :);        % dx(1)...dx(N) from dx0
Gamma_pred = Gamma(1:nx*N, :);      % dx(1)...dx(N) from du

% C applied to each predicted state: C_blk * [dx(1);...;dx(N)]
C_blk = kron(eye(N), C);  % N*ny x N*nx

% C_blk * Gamma_pred maps du -> predicted outputs
CG = C_blk * Gamma_pred;  % N*ny x N*nu
CP = C_blk * Phi_pred;    % N*ny x nx

% Slack selector: maps eps part of z to N*ny vector
I_eps = eye(N*ny);

% Number of inequality constraints:
%   2*N*nu (input bounds) + 2*N*ny (output bounds) + N*ny (slack >= 0)
n_ineq = 2*N*nu + 2*N*ny + N*ny;

% Build Ain (w.r.t. z = [du; eps])
Ain = zeros(n_ineq, nz);
bin_const = zeros(n_ineq, 1);  % constant part (independent of dx0, x_ss, u_ss)

% We'll also need matrices to compute bin from (dx0, x_ss, u_ss) at runtime:
% bin = bin_const + Bin_dx0 * dx0 + Bin_xss * x_ss + bin_uss * u_ss
Bin_dx0 = zeros(n_ineq, nx);
Bin_xss = zeros(n_ineq, nx);
Bin_uss = zeros(n_ineq, nu);

row = 0;

% (a) Input upper: du(k) <= u_max - u_ss  for k=1..N
Ain(row+1:row+N*nu, 1:N*nu) = eye(N*nu);
bin_const(row+1:row+N*nu) = repmat(u_max, N*nu, 1);
Bin_uss(row+1:row+N*nu, :) = repmat(-eye(nu), N, 1);
row = row + N*nu;

% (a) Input lower: -du(k) <= -u_min + u_ss  for k=1..N
Ain(row+1:row+N*nu, 1:N*nu) = -eye(N*nu);
bin_const(row+1:row+N*nu) = repmat(-u_min, N*nu, 1);
Bin_uss(row+1:row+N*nu, :) = repmat(eye(nu), N, 1);
row = row + N*nu;

% (b) Output upper: CG*du - I*eps <= y_max - CP*dx0 - C_blk_xss
%     where C_blk_xss = C*x_ss repeated N times
Ain(row+1:row+N*ny, 1:N*nu)        = CG;
Ain(row+1:row+N*ny, N*nu+1:end)    = -I_eps;
bin_const(row+1:row+N*ny)           = repmat(y_max, N*ny, 1);
Bin_dx0(row+1:row+N*ny, :)          = -CP;
Bin_xss(row+1:row+N*ny, :)          = repmat(-C, N, 1);
row = row + N*ny;

% (b) Output lower: -CG*du - I*eps <= -y_min + CP*dx0 + C_blk_xss
Ain(row+1:row+N*ny, 1:N*nu)        = -CG;
Ain(row+1:row+N*ny, N*nu+1:end)    = -I_eps;
bin_const(row+1:row+N*ny)           = repmat(-y_min, N*ny, 1);
Bin_dx0(row+1:row+N*ny, :)          = CP;
Bin_xss(row+1:row+N*ny, :)          = repmat(C, N, 1);
row = row + N*ny;

% (c) Slack non-negativity: -eps(k) <= 0
Ain(row+1:row+N*ny, N*nu+1:end) = -I_eps;
bin_const(row+1:row+N*ny) = 0;
row = row + N*ny;

% quadprog options
qp_opts = optimoptions('quadprog', 'Display', 'off');

fprintf('MPC QP formulated (N = %d, %d decision vars, %d constraints)\n', ...
    N, nz, n_ineq);

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

    % D. MPC via quadprog
    dx0 = x_hat_k - x_ss_k;  % initial deviation

    % Linear cost term
    f_vec = [f_du_matrix * dx0; zeros(N*ny, 1)];

    % RHS of inequality constraints
    bin = bin_const + Bin_dx0 * dx0 + Bin_xss * x_ss_k + Bin_uss * u_ss_k;

    % Solve QP: min 0.5*z'*H*z + f'*z  s.t. Ain*z <= bin
    [z_opt, ~, exitflag] = quadprog(H, f_vec, Ain, bin, [], [], [], [], [], qp_opts);

    if exitflag == 1
        du_opt    = z_opt(1:nu);          % first input move
        slack_now = z_opt(N*nu+1:N*nu+ny); % first slack
    else
        warning('MPC infeasible at t = %d (exitflag %d). Applying u_ss.', t, exitflag);
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

sgtitle('Offset-Free MPC with Augmented Disturbance Observer', ...
    'FontSize', 14, 'FontWeight', 'bold');
