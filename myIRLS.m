
function [beta, history] = myIRLS(X, y, lambda, beta0, tol, maxIter, penaltyFunction)

% initialize variables
n = size(X, 2);
iter = 0;
err = inf;
w = eye(n);
history = NaN(maxIter, 2);

% loop until convergence or max iteration
while err > tol && iter < maxIter
    
    if iter == 0
        betaOld = beta0;
    else
        betaOld = betaNew;
    end
    
    for i = 1:n
        w(i, i) =  penaltyFunction(i, betaOld);
    end
    
    % solve for beta
    betaNew = (X' * X + lambda * (w' * w)) \ (X' * y);
    
    % calculate error
    err = norm(betaNew - betaOld);
    % book keeping for ouput
    iter = iter + 1;
    history(iter, 1) = iter;
    history(iter, 2) = err;
end
history = history(1:iter, :);
beta = betaNew;
end