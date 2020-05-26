
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
