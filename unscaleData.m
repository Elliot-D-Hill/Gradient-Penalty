
% unscale design matrix after scaling by some norm (see file scaleData.m)
function beta = unscaleData(betaTilda, invS)
    beta = invS * betaTilda;
end