
function [X, xGradient, mRow, nCol, f] = createDesignMatrix(nBasis, xData)

% basis functions
n = 0:nBasis;
f = @(t, n) t.^(0:nBasis);
fp = @(t, n) n .* t.^abs(n-1);

% design matrix
X = f(xData, n);

% gradient of design matrix
xGradient = fp(xData, n);

% dimensions of X
mRow = size(xData, 1);
nCol = size(X, 2);
end
