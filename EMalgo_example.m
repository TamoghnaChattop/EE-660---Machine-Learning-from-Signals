clearvars;
close all;

%% All samples and underlying distributions 
% comment/ uncomment whichever you want to test

N = 100;
nc = 2; % number of classes
real_mu1 = -1;
real_mu2 = 1;
real_sigma1 = 1; % std dev
real_sigma2 = 2; % std dev

% Normal underlying distribution
X1 = normrnd(real_mu1,real_sigma1, [N, 1]);
X2 = normrnd(real_mu2,real_sigma2, [N, 1]);
minX = min([X1;X2]);
maxX = max([X1;X2]);
range = (minX-0.01):0.01:(maxX+0.01);
real_pdf1 = normpdf(range, real_mu1, real_sigma1);
real_pdf2 = normpdf(range, real_mu2, real_sigma2);

% Uniform underlying distribution
% a1 = real_mu1 - sqrt(3)*real_sigma1;
% b1 = real_mu1 + sqrt(3)*real_sigma1;
% X1 = unifrnd(a1, b1, [N 1]);
% a2 = real_mu2 - sqrt(3)*real_sigma2;
% b2 = real_mu2 + sqrt(3)*real_sigma2;
% X2 = unifrnd(a2, b2, [N 1]);
% range = (min(a1, a2)-0.1):0.01:(max(b1, b2)+0.1);
% real_pdf1 = unifpdf(range, a1, b1);
% real_pdf2 = unifpdf(range, a2, b2);

fig = figure; hold on; grid;
scatter(X1, zeros(N,1), 'blue');
scatter(X2, zeros(N,1), 'red');
plot(range, real_pdf1, 'blue', 'LineWidth', 1.5);
plot(range, real_pdf2, 'red', 'LineWidth', 1.5);

%% Estimating with all samples
all_sample_mean1 = mean(X1);
all_sample_mean2 = mean(X2);
all_sample_var1 = var(X1);
all_sample_var2 = var(X2);
full_est_norm1 = normpdf(range, all_sample_mean1, real_sigma1);
full_est_norm2 = normpdf(range, all_sample_mean2, real_sigma2);

plot(range, full_est_norm1, '--b', 'LineWidth', 1.5);
plot(range, full_est_norm2, '--r', 'LineWidth', 1.5);

%% Create single partially labeled dataset
nl = 5;
nu = N-nl;

labeled = [randperm(N, nl), N+randperm(N, nl)];
y = zeros(2*N, 1);
y(labeled) = [ones(nl,1); 2*ones(nl,1)];

%% EM algorithm
X = [X1; X2];
[mu, sigma, pi] = simpleEM_GMM(X, y, nc);

%% Compare output to real distribution and to all labeled training set
figure(fig); hold on;
est_norm1 = normpdf(range, mu(1), sigma(1));
est_norm2 = normpdf(range, mu(2), sigma(2));
plot(range, est_norm1, '--c', 'LineWidth', 1.5);
plot(range, est_norm2, '--m', 'LineWidth', 1.5);

legend({'Samples class 1', 'Samples class 2', 'True dist class 1', 'True dist class 2', ...
    'Est dist from all samples class 1', 'Est dist from all samples class 2', ...
    'Est dist EM class 1', 'Est dist EM class 2'})
