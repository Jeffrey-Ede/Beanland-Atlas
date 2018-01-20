function [ r_and_p ] = pearson_r_and_p(x, y)
%Pearson r and p between 2 data sets
%
%x: One of the data sets
%y: The other data set

[R, P] = corrcoef(x, y);
r_and_p = [R(1, 2) P(1, 2)];
end