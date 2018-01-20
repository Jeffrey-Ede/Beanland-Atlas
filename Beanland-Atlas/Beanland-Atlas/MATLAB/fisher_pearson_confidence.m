function interval = fisher_pearson_confidence(rho, n, confidence)
%Use the Fisher transform to calculate the confidence interval for a
%Pearson normalised product moment correlation coefficient
%
%rho: Pearson coefficient
%n: number of samples
%confidence: Confidence to get interval for e.g. 0.95

interval = zeros(2, 1);
ste = 1/sqrt(n-3);
ci = norminv(confidence);
interval(2) = tanh(atanh(rho) + ci*ste);
interval(1) = tanh(atanh(rho) - ci*ste);
end