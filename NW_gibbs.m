clear;
close all;
clc;
rng(1);

%% DGP

lowerbound = 15;
upperbound = 80;
numbers = 1000;

sigma_GDP = 1;
sigma_kernel = 0.5;

% generate data
observation = randi([lowerbound upperbound], 1, numbers)';
order_grid = sort(observation);

sym x;
sym b;

alpha = 0.55;
bandwidth_vector = [];
y_vector = [];

% define true g function
g = @(x) 3 * (x).^(1 - alpha)/(1 - alpha) - x.^(4/7) - 13;

for index = 1 : length(observation)
    y = observation(index);
    y_vector(index, 1) = g(y) + normrnd(0, sigma_GDP);
end

data = [observation y_vector];

%% Estimation

v = unique(observation);
m = length(v);
Q = double((observation == v'));
h(2 : m) = v(2 : m) - v(1 : m - 1);
H = eye(m);

for t = 3 : m
    H(t,t-2) = h(t)/h(t-1);
    H(t,t-1) = -(1+h(t)/h(t-1));
end

n = length(observation);
g10 = 0;

% determines the intercept and
% slope of the line representing prior mean of g
g20 = 1;

gtilde = [g10 g20 zeros(1, m - 2)]';
g0 = H \ gtilde;
G0 = 100 * eye(2);
Sigma = diag([0 0 h(3 : end)]);
Sigma(1 : 2, 1 : 2) = G0;
K = H' / Sigma * H;

nu0 = 0.1;
delta0 = 0.1;
alpha0 = 0.1;
gamma0 = 0.1;

iteration = 15000;
burn = 5000;
post = iteration - burn;
gs = NaN(m,post);
sig_square_vector = NaN(1, post);
tau_square_vector = NaN(1, post);
QQ = Q' * Q;
Qy = Q' * y_vector;

% smoothing parameter (10 or 0.01)
tau_suqare = 0.01;

% initialize sampler
sig_square = 1;

% plot the true function
figure(2)
sc = 5;
gx = g(grid);
plot(grid, gx, 'linewidth', 2, 'color', 'b');
hold on;
scatter(data(:, 1), y_vector, sc, 'filled', 'k'); %

% update posterior
for iter = 1:iteration

    Ghat_gibbs = inv(K / tau_suqare + QQ / sig_square);
    ghat_gibbs = Ghat_gibbs * (K / tau_suqare * g0 + Qy / sig_square);

    Ghat_gibbs = (Ghat_gibbs + Ghat_gibbs') / 2;
    g_random = mvnrnd(ghat_gibbs, Ghat_gibbs)';

    sig_square = iwishrnd(delta0 + (y_vector - Q * g_random)' * ...
        (y_vector - Q * g_random), nu0 + n);

    %tau_suqare = iwishrnd(gamma0 + (g_random - g0)' * ...
        %K * (g_random - g0), alpha0 + m);

    if iter > burn
        gs(:, iter - burn) = g_random;
        sig_square_vector(iter - burn) = sig_square;
        tau_square_vector(iter - burn) = tau_suqare;
    end
end

% plot the graph
plot(v, mean(gs, 2), 'linewidth', 2, 'color', 'r');
plot(v, prctile(gs, 97.5, 2), '--', 'linewidth', 1, 'color', 'm');
plot(v, prctile(gs, 2.5, 2), '--', 'linewidth', 1, 'color', 'm');
legend('g(s)', 'data', 'posterior mean', ...
    'posterior 97.5th percentile', 'posterior 2.5th percentile', ...
    'location', 'best');
title('\pi(g|y)');
%axis([15 80, 3 25])
axis([20 30, 6 11.5])
xlabel('age (from 20-30)') %15-80 or 20-30
ylabel('wage (ten thousand dollars)')
