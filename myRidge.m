function beta = myRidge(X, y, lambda, normalizeData)
    
    if normalizeData == true
            X = normalize(X);
    end 

    XTX = X' * X;
    XTy = X' * y;
    beta = (XTX + lambda * eye(length(XTX))) \ XTy;
end