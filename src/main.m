clear all
close all
clc
%rng(2077333712)

nodeCounts = [10, 10, 7];
layerCount = numel(nodeCounts) - 1;

weightMatrix = zeros();

for i = 1:layerCount
    n = nodeCounts(i);
    m = nodeCounts(i + 1);
    weightMatrix(1:n,1:m,i) = randomWeights(n, m);
end