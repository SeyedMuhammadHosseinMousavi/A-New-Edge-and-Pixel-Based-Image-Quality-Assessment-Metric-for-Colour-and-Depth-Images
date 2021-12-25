function pcc = compute_PearsonCorrelationCoefficient (signal1, signal2)

originalRowSize = size(signal1,1);
originalColSize = size(signal1,2);

signal1 = signal1(:);
signal2 = signal2(:);

mean_signal1 = sum(signal1)/numel(signal1);
signal1 = signal1 - mean_signal1;

mean_signal2 = sum(signal2)/numel(signal2);
signal2 = signal2 - mean_signal2;

pcc = sum(signal1.*signal2)/ (std(signal1)*std(signal2));

end

