function [mae] = meanAbsoluteError(signal1, signal2)

originalRowSize = size(signal1,1);
originalColSize = size(signal1,2);

signal1 = signal1(:);
signal2 = signal2(:);

mae = sum(abs(signal1 - signal2))/(originalRowSize*originalColSize);

end

