
%% ------ Rugularizing boundary oscillations ------

% This program tests a novel gradient-based penalty method that 
% regularization polynomial regression models fit to functions 
% that have pathological oscillations at their boundary.

%% ------ Setup environment ------

clc; close all; clear;
format short
rng(10); % set random seed for reproducability

%% ------ Program parameters ------

problem = 'runge'; % choose from: 'runge', 'step'
penalty = 'derivative'; % choose from: 'pNorm', 'derivative'
priorChoice = 'derivative'; % choose from: 'none', 'derivative'

pNorm = 1; % norm of penalty function
lambdas = [0.1.^(1:15), 0]; % regularization parameter
kFolds = 10; % number of cross validation folds
scaleNorm = 2; % Norm used to scale data
nDegree = 15; % highest polynomial degree for basis functions
nData = 31; % number of sample points

% iteratively reweighted least squares parameters
% parameter optimization stopping criteria
tol = 10^-9; 
maxIter = 1000;
% prevents division by zero
epsilon1 = 10^-4;
epsilon2 = 10^-4; 

%% ------ Data generation ------

xData = linspace(-1, 1, nData)'; % uniform sampling in (-1, 1)
yDataStd = 0.0; % magnitude of the noise

switch problem
    case 'runge' % Runge function
        yData = 1.0 ./ (1 + 25 * xData.^2) + randn(size(xData,1), 1) * yDataStd; 
    case 'step' % Step function
        yData = stepFunction(xData) + randn(size(xData,1), 1) * yDataStd;
end

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

%% ------ Cross-validation grid search ------

% initialize MSE and cross-validation MSE to 0
MSECV = zeros(nDegree, length(lambdas));
MSE = zeros(nDegree, length(lambdas));

% for each polynomial degree
for degree = 1:nDegree
    
    % setup design matrix
    [X, xGradient, ~, nCol] = createDesignMatrix(degree, xData);
    
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
    
    % for each value of the regularization parameter, lambda
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


%% ------ Plot results ------

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

[X, xGradient, mRow, nCol, f] = createDesignMatrix(minDegreeIdx, xData);

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
[betaBest, ~] = myIRLS(X, yData, lambdas(minLambdaIdx), beta0, tol, maxIter, penaltyFunction);

% plot prior
figure
subplot(1,3,1)
plot(prior)
set(gca,'FontSize',14)
ylabel('$\tilde{y}$''', 'Interpreter', 'latex')
xlabel('x')
subplot(1,3,2)
plot(xGradient * betaBest)
set(gca,'FontSize',14)
ylabel('$G\beta$', 'Interpreter', 'latex')
xlabel('x')
subplot(1,3,3)
plot(abs(prior - xGradient * betaBest))
set(gca,'FontSize',14)
ylabel('$|\tilde{y}'' - G\beta|$', 'Interpreter', 'latex')
xlabel('x')

% plot fit with smallest cross-validation MSE
betaBest = unscaleData(betaBest, invS);
figure
xPlot = (min(xData) : 0.01 : max(xData))';
plot(xData, yData, 'ro', 'displayname', 'data', 'linewidth', 2, 'markersize', 7)
hold on
plot(xPlot, f(xPlot, 0:minDegreeIdx) * betaBest, 'displayname', ['Fit: ' num2str(beta')],'linewidth',2)
legend('Data', 'Fit')
ylabel('y')
xlabel('x')
set(gca,'FontSize',14)
