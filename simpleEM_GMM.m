function [mu, sigma, pi] = simpleEM_GMM(X, y, nc)
%simpleEM_GMM Example of EM algorithm for 1D GMM

% Number of samples and features
[N, D] = size(X);
if D>1
    error('Number of features greater than 1; not supported')
end

% Range of features (used for ploting)
range = min(X):0.01:max(X);

% Plot labeled and unlabeled samples
nu = sum(y==0);
nl = N - nu;
figure; hold on; grid;
title('Estimated Gaussians at each EM iteration')
scatter(X(y==0), zeros(nu, 1),[], [0 0.7 0]);
colors = {'b', 'r', 'y', 'm', 'c', 'k'};

for j = 1:nc
    color = colors{rem(j, length(colors))};
    labeledX = X(y==j);
    scatter(labeledX, zeros(length(labeledX), 1), [], color, 'filled');    
end

% Get MLE from labeled data
% Note that the code below only works properly because D = 1;
mu = zeros(nc, 1);
sigma = zeros(nc, 1);
pi = zeros(nc, 1);
for j = 1:nc
    labeledX = X(y==j);
    mu(j) = mean(labeledX);
    sigma(j) = sqrt((labeledX-mu(j))'*(labeledX-mu(j))/nl);
    pi(j) = size(labeledX, 1)/nl;
    
    % Plot the curve
    normcurve = normpdf(range, mu(j), sigma(j));
    color = colors{rem(j, length(colors))};
    plot(range, normcurve, [':' color], 'LineWidth', 1.5);
end

k = 1;
maxIter = 100;
epsilon = 0.01;
ll_update = 1;

% Log likelihood
ll = zeros(maxIter, 1);
for i = 1:length(X)
    if y(i)~=0
        class = y(i);
        ll(1) = ll(1) + log(pi(class)*normpdf(X(i), mu(class), sigma(class)));
    else
        temp = 0;
        for j = 1:nc
            temp = temp + pi(j)*normpdf(X(i), mu(j), sigma(j));
        end
        ll(1) = ll(1) + log(temp);
    end
end

while k<maxIter && ll_update>epsilon
    % E-step
    gamma = zeros(length(X), nc);
    for i = 1:length(X)
        for j = 1:nc
            if y(i) == 0
            gamma(i, j) = pi(j)*normpdf(X(i), mu(j), sigma(j));
            elseif y(i) == j
                gamma(i, j) = 1;
            end
        end
        gamma(i, :) = gamma(i, :)/sum(gamma(i, :));
    end
    
    % M-step
    l = zeros(1, nc);
    for j = 1:nc
        l(j) = sum(gamma(:, j));
        mu(j) = gamma(:, j)'*X/l(j);
        sigma(j) = sqrt((gamma(:,j).*(X - mu(j)))'*(X-mu(j))/l(j));
        pi(j) = l(j)/N;
    end
    k = k + 1;

    for i = 1:length(X)
        if y(i)~=0
            class = y(i);
            ll(k) = ll(k) + log(pi(class)*normpdf(X(i), mu(class), sigma(class)));
        else
            temp = 0;
            for j = 1:nc
                temp = temp + pi(j)*normpdf(X(i), mu(j), sigma(j));
            end
            ll(k) = ll(k) + log(temp);
        end
    end 
    ll_update = ll(k) - ll(k-1);
    if ll_update<0 % Sanity check
        error('Something wrong: log likelihood decreased.');
    end
    
    % Visualization
    norm1 = normpdf(range, mu(1), sigma(1));
    norm2 = normpdf(range, mu(2), sigma(2));

    plot(range, norm1, 'blue');
    plot(range, norm2, 'red');
end

if (k == maxIter)
    warning('EM algorithm reached maximum number of iterations before converging.');
else
    fprintf('EM algorithm converged after %d iterations.\n', k);
end

legend('Unclassified samples', 'Samples class 1', 'Samples class 2', ...
    'Est dist using only labeled samples class 1', 'Est dist using only labeled samples class 2',...
    'Est dist at each EM iteration class 1', 'Est dist at each EM iteration class 2');

figure; hold on;
title('Log Likelihood vs EM iterations')
plot(1:k, ll(1:k)); grid on;
ylabel('Log likelihood')
xlabel('Iterations of EM')

end

