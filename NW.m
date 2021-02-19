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

% define grid
grid = linspace(15, 80, 261);

% gaussian kernel
kernel = @(b) ((2 * pi)*sigma_kernel^2)^(-1/2) * ...
    exp(-(b - 0)^2/(2 * sigma_kernel^2));

% first bandwidth is the optimal bandwidth
bandwidth_vector(3, :) =  10;
bandwidth_vector(1, :) =  1.059 * sigma_kernel * numbers^(-1/5);
bandwidth_vector(2, :) =  40;

% compute g_hat
ghat_1 = [];
ghat_2 = [];
ghat_3 = [];

for outerloop = 1 : length(grid)

    ghat_numerator = 0;
    ghat_denominator = 0;

    for innerloop = 1 : numbers
        ghat_numerator = ghat_numerator + ...
            kernel((grid(outerloop) - ...
            data(innerloop, 1)) / bandwidth_vector(1)) * ...
            data(innerloop, 2);

        ghat_denominator = ghat_denominator + ...
            kernel((grid(outerloop) - ...
            data(innerloop, 1)) / bandwidth_vector(1));

    end
    ghat_1(outerloop) = ghat_numerator / ghat_denominator;
end

for outerloop = 1 : length(grid)

    ghat_numerator = 0;
    ghat_denominator = 0;

    for innerloop = 1 : numbers
        ghat_numerator = ghat_numerator + ...
            kernel((grid(outerloop) - ...
            data(innerloop, 1)) / bandwidth_vector(2)) * ...
            data(innerloop, 2);

        ghat_denominator = ghat_denominator + ...
            kernel((grid(outerloop) - ...
            data(innerloop, 1)) / bandwidth_vector(2));

    end
    ghat_2(outerloop) = ghat_numerator / ghat_denominator;
end

for outerloop = 1 : length(grid)

    ghat_numerator = 0;
    ghat_denominator = 0;

    for innerloop = 1 : numbers
        ghat_numerator = ghat_numerator + ...
            kernel((grid(outerloop) - ...
            data(innerloop, 1)) / bandwidth_vector(3)) * ...
            data(innerloop, 2);

        ghat_denominator = ghat_denominator + ...
            kernel((grid(outerloop) - ...
            data(innerloop, 1)) / bandwidth_vector(3));

    end
    ghat_3(outerloop) = ghat_numerator / ghat_denominator;
end

% compute data points of the real function
real_function = [];
for i = 1 : length(grid)
    real_function(i, :) = g(grid(i));
end

% plot
figure(1)
plot(grid, ghat_3,'linewidth', 2,'color', 'g')
hold on;
plot(grid, ghat_2,'linewidth', 2, 'color', 'b')
hold on;
plot(grid, ghat_1,'linewidth', 2, 'color', 'r')
hold on;
plot(grid, real_function, '--','linewidth', 2, 'color', 'k')
axis([15 80, 3 24])

legend('bandwidth=5', 'bandwidth=30', 'Silverman bandwidth', 'true function')
xlabel('age (from 15-80)')
ylabel('wage (ten thousand dollars)')
