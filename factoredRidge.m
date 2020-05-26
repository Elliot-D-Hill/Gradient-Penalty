
function beta = factoredRidge(X, y, lambda, normalizeData)
    
    if normalizeData == true
            X = normalize(X);
    end 
    
    XXT = X * X';
    eta = (XXT + lambda * eye(length(XXT))) \ y;
    beta = X' * eta;
end
