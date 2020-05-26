
%% ------ Rugularizing the Runge function ------

clc; close all; clear;
format short
rng(10); % set random seed for reproducability

%% ------ program parameters ------

penalty = 'derivative'; % choose from: 'pNorm', 'derivative'
priorChoice = 'derivative'; % choose from: 'none', 'derivative'
pNorm = 1; % norm of penalty function
lambdas = [0.1.^(1:15), 0]; % regularization parameter
kFolds = 10; % number of cross validation folds
scaleNorm = 2; % Norm used to scale data
nDegree = 15; % highest degree of polynomial basis functions
nData = 31; % number of sample points

%% ------ data generation ------

xData = linspace(-1, 1, nData)'; % uniform sampling in (-1, 1)
yDataStd = 0.0; % magnitude of the noise
yData = 1.0 ./ (1 + 25 * xData.^2) + randn(size(xData,1), 1) * yDataStd; % Runge function

%% ------ Prior ------
switch priorChoice
    case 'none'
        prior = zeros(nData, 1);
    case 'derivative'
        dx = xData(1) - xData(2);
        [dydx] = finiteDifference(yData, dx);
        prior = dydx;
        prior = prior / norm(prior, 2);
end

% iteratively reweighted least squares parameters
tol = 10^-9;
maxIter = 1000;
epsilon1 = 10^-4;
epsilon2 = 10^-4;

% initialize MSE and cross-validation MSE to 0
MSECV = zeros(nDegree, length(lambdas));
MSE = zeros(nDegree, length(lambdas));

for degree = 1:nDegree
    
    % setup design matrix
    [X, xGradient, ~, nCol] = designMatrix(degree, xData);
    
    % normalize data
    xTemp = X(:,2:end); % don't need to normalize the intercept
    [xTemp, yData, ~, ~] = scaleData(xTemp, yData, scaleNorm);
    X = [X(:,1) xTemp];
    
    % normalize gradient data
    gradTemp = xGradient(:,2:end);
    [gradTemp, ~, ~, ~] = scaleData(gradTemp, yData, scaleNorm);
    xGradient = [xGradient(:,1) gradTemp];
    
    % split data into k cross validation folds
    [xFolds, yFolds] = createFolds(X, yData, kFolds);
    
    % initial guess
    beta0 = zeros(nCol, 1);
    
    % penalty function and prior setup
    switch penalty
        case 'pNorm'
            penaltyFunction = @(idx, betaOld) pNormPenalty(idx, betaOld, pNorm, epsilon1);
        case 'derivative'
            penaltyFunction = @(idx, betaOld) derivativePenalty(idx, betaOld, xGradient, prior, pNorm, epsilon1, epsilon2);
    end
    
    for lambda = 1:length(lambdas)
        for kthFold = 1:kFolds
            
            % split data into training and test set
            [xTrain, yTrain, xTest, yTest] = splitTrainTest(xFolds, yFolds, kthFold);
            
            % fit training data
            [beta, ~] = myIRLS(xTrain, yTrain, lambdas(lambda), beta0, tol, maxIter, penaltyFunction);
            
            % calculate cross validation MSE for test set
            yHat = xTest * beta;
            MSECV(degree, lambda) = MSECV(degree, lambda) + sum((yTest - yHat).^2);
        end
        
        [beta1, ~] = myIRLS(X, yData, lambdas(lambda), beta0, tol, maxIter, penaltyFunction);
        MSE(degree, lambda) = MSE(degree, lambda) + sum((yData - X * beta1).^2);
    end
end

MSECV = MSECV ./ kFolds;
MSE = MSE ./ nData;

%% ------ plot results ------

set(gca,'FontSize',14)

% plot CV error and MSE
figure(1)
subplot(1,2,1)
degrees = (1:nDegree)';
nLambdas = (1:length(lambdas))';
pcolor(nLambdas, degrees, log(MSE));
c = colorbar;
c.Label.String = 'log MSE';
c.Label.FontSize = 16;
xticks(1:2:size(lambdas,2))
xticklabels(lambdas(1:2:end));
xlabel('Lambda', 'FontSize', 16);
ylabel('Degree of fit', 'FontSize', 16);


subplot(1,2,2)
[nLambdas, degrees] = meshgrid(nLambdas, degrees);
pcolor(nLambdas, degrees, log(MSECV));
c = colorbar;
c.Label.String = 'log MSE_{CV}';
c.Label.FontSize = 16;
xticks(1:2:size(lambdas,2))
xticklabels(lambdas(1:2:end));
xlabel('Lambda', 'FontSize', 16);
ylabel('Degree of fit', 'FontSize', 16);


% get lambda and degree at minimum MSE CV
minMatrix = min(MSECV(:));
[row,col] = find(MSECV==minMatrix);

[minMSECV, ~] = min(MSECV, [], 'all', 'linear');
[minDegreeIdx, minLambdaIdx] = find(MSECV==minMSECV);

[X, xGradient, mRow, nCol, f] = designMatrix(minDegreeIdx, xData);

% normalize data
[X, yData, invS, yRho] = scaleData(X, yData, scaleNorm);

gradTemp = xGradient(:,2:end);
[gradTemp, ~, ~, ~] = scaleData(gradTemp, yData, scaleNorm);
xGradient = [xGradient(:,1) gradTemp];

% initial guess
beta0 = zeros(nCol, 1);
% penalty function and prior setup
switch penalty
    case 'pNorm'
        penaltyFunction = @(idx, betaOld) pNormPenalty(idx, betaOld, pNorm, epsilon1);
    case 'derivative'
        penaltyFunction = @(idx, betaOld) derivativePenalty(idx, betaOld, xGradient, prior, pNorm, epsilon1, epsilon2);
end

% fit model using degree and lambda that gave minimum CV error
[betaBest1, ~] = myIRLS(X, yData, lambdas(minLambdaIdx), beta0, tol, maxIter, penaltyFunction);
lambdas(minLambdaIdx)

figure(5)
subplot(1,3,1)
plot(prior)
set(gca,'FontSize',14)
ylabel('$\tilde{y}$''', 'Interpreter', 'latex')
xlabel('x')
subplot(1,3,2)
plot(xGradient * betaBest1)
set(gca,'FontSize',14)
ylabel('$G\beta$', 'Interpreter', 'latex')
xlabel('x')
subplot(1,3,3)
plot(abs(prior - xGradient * betaBest1))
set(gca,'FontSize',14)
ylabel('$|\tilde{y}'' - G\beta|$', 'Interpreter', 'latex')
xlabel('x')

sum(abs(prior - xGradient * betaBest1))

betaBest1 = unscale(betaBest1, invS);

figure(4)

xPlot = (min(xData) : 0.01 : max(xData))';
plot(xData, yData, 'ro', 'displayname', 'data', 'linewidth', 2, 'markersize', 7)
hold on
plot(xPlot, f(xPlot, 0:minDegreeIdx) * betaBest1, 'displayname', ['Fit: ' num2str(beta')],'linewidth',2)
legend('Data', 'Fit')
ylabel('y')
xlabel('x')
set(gca,'FontSize',14)

%% ------ Penalty functions ------

function w = pNormPenalty(idx, betaOld, p, epsilon)
w =  1 / (abs(betaOld(idx))^(-(1/2) * p + 1) + epsilon);
end

function w = derivativePenalty(idx, betaOld, xGradient, prior, p, epsilon1, epsilon2)
g = xGradient * betaOld;
r = prior(idx) - g(idx);
w =  (sqrt(abs(r)^p) + epsilon1) / (abs(r) + epsilon2);
end

% create folds for k-folds cross validation
function [xFolds, yFolds] = createFolds(X, y, kthFold)

tempData = [X y];
randomRowsData = tempData(randperm(size(tempData, 1)), :);

randomRowsX = randomRowsData(:, 1:end-1);
randomY = randomRowsData(:, end);

mRow = size(randomRowsX, 1);
nCol = size(randomRowsX, 2);

r = diff(fix(linspace(0, mRow, kthFold + 1)));
xFolds = mat2cell(randomRowsX, r, nCol);
yFolds = mat2cell(randomY, r, 1);
end

function [xTrain, yTrain, xTest, yTest] = splitTrainTest(xFolds, yFolds, kthFold)

% set kth fold as test set
xTest = cell2mat(xFolds(kthFold));
yTest = cell2mat(yFolds(kthFold));

% remove kth fold
xFolds(kthFold) = [];
yFolds(kthFold) = [];

% training k - 1 folds
xTrain = cell2mat(xFolds);
yTrain = cell2mat(yFolds);
end

function [X, xGradient, mRow, nCol, f] = designMatrix(nBasis, xData)

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


function [xTilda, yTilda, invS, yRho] = scaleData(X, y, normChoice)

    nCol = size(X, 2);
    
    % create diagonal matrix of scaling factors for each column of X
    invS = zeros(nCol, nCol);
    
    for k = 1:nCol
        invS(k, k) = 1 / norm(X(:,k), normChoice);
    end
  
    % create scaling factor for y
    yRho = 1 / norm(y, normChoice);
    % scale X matrix
    xTilda = yRho * X * invS;
    % scale y vector
    yTilda = yRho * y;
end

function beta = unscale(betaTilda, invS)
    beta = invS * betaTilda;
end

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
