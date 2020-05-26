
% Creates weights for iteratively rewieghted least squares
% Weights depend on the gradient of the design matrix and a prior
function w = derivativePenalty(idx, betaOld, xGradient, prior, p, epsilon1, epsilon2)
g = xGradient * betaOld;
r = prior(idx) - g(idx);
w =  (sqrt(abs(r)^p) + epsilon1) / (abs(r) + epsilon2);
end