
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