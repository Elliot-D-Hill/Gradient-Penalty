
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