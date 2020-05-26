
% second order finite difference approximation of the gradient
function [dydx] = finiteDifference(yData, dx)

yDiff = NaN(length(yData), 1);
dydx = NaN(length(yData), 1);

% interior values
for i = 2:length(yData) - 1
    yDiff(i) = yData(i-1) - yData(i+1);
    dydx(i) =  yDiff(i) / (2 * dx);
end

% boundary values
dydx(1) = (3 * yData(1) - 4 * yData(2) + yData(3)) / (2 * dx);
dydx(end) = (-3 * yData(end) + 4 * yData(end-1) -yData(end-2)) / (2 * dx);
end
