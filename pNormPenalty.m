
% Creates weights for iteratively rewieghted least squares
% p = 1 gives lasso penalty, p = 2 gives ridge penalty
% This function assumes no prior
function w = pNormPenalty(idx, betaOld, p, epsilon)
w =  1 / (abs(betaOld(idx))^(-(1/2) * p + 1) + epsilon);
end