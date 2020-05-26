
function y = stepFunction(t)
y = NaN(length(t), 1);
for i = 1:length(t)
    if t(i) >= 0 % output 1 if the argument is greater than or equal to 0
        y(i) = 1;
    else
        y(i) = 0; % default output value is 0
    end
end
end