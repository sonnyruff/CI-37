function weightMatrix = generateWeightMatrix( nodeCounts )
%GENERATEWEIGHTMATRIX Summary of this function goes here
%   Detailed explanation goes here
    layerCount = numel(nodeCounts) - 1;
    
    weightMatrix = zeros();
    
    for i = 1:layerCount
        m = nodeCounts(i + 1);
        n = nodeCounts(i);
        weightMatrix(1:m,1:n,i,1) = randomWeights(m, n);
    end
end

