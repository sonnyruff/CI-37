function [ weightMatrix ] = generateWeightMatrix( nodeCounts )
%GENERATEWEIGHTMATRIX Summary of this function goes here
%   Detailed explanation goes here
    layerCount = numel(nodeCounts) - 1;
    
    weightMatrix = zeros();
    
    for i = 1:layerCount
        n = nodeCounts(i);
        m = nodeCounts(i + 1);
        weightMatrix(1:n,1:m,i) = randomWeights(n, m);
    end
end

