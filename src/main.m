clear
close all
clc
%rng(2077333712)

%%inputFile = dlmread("../data/features.txt");
%%desiredOutputFile = dlmread("../data/targets.txt");
inputFile = [0, 0; 1, 0; 0, 1; 1, 1];
desiredOutputs = [0, 1, 1, 0];
values = zeros();
error = zeros();
errorSumSquared = zeros();

% Learning rate
a = 0.2;

%%nodeCounts = [10, 9, 11, 7];
nodeCounts = [2, 2, 1];
layerCount = numel(nodeCounts);

% Generate a matrix with random weights for all the nodes in the network
weightMatrix = generateWeightMatrix(nodeCounts);

iteration = 1;
solutionFound = 0;

% Iterate over every entry in the input file
while ~solutionFound & iteration < 10000
    inputNum = mod(iteration, size(inputFile, 1)) + 1;
    values(1,1:nodeCounts(1)) = inputFile(inputNum,:);
    
    % Iterate FORWARD over every layer in the network
    for layer = 2:layerCount
        prevL = layer - 1;
        prevLSize = nodeCounts(prevL);
        
        % Iterate over every node in a layer
        for node = 1:nodeCounts(layer)
            % Truncate the inputs and outputs
            currInputs = values(prevL, 1:prevLSize);
            currWeights = weightMatrix(node, 1:prevLSize, prevL, iteration);
            values(layer, node) = perceptron(currInputs, currWeights, -1);
        end
    end
    
    %%desiredOutputs = full(vec2mat(ind2vec(desiredOutputFile(inputNum))',nodeCounts(end)));
    %%error(1:nodeCounts(end)) = desiredOutputs - values(1:nodeCounts(end));
    error(1:nodeCounts(end)) = desiredOutputs(inputNum) - values(1:nodeCounts(end));
    errorSumSquared(inputNum) = sumsqr(error(1:nodeCounts(end)));
    
    % Iterate BACKWARD over every layer in the network
    for layer = layerCount:-1:2
        prevL = layer - 1;
        nextL = layer + 1;
        currLSize = nodeCounts(layer);
        prevLSize = nodeCounts(prevL);
        currLValues = values(layer, 1:currLSize);
        prevLValues = values(prevL, 1:prevLSize);
        
        if layer ~= layerCount
            error(1:currLSize) = sum(errorGrad(nextL, 1:nodeCounts(nextL)));
        end
        errorGrad(layer, 1:currLSize) = currLValues .* (1 - currLValues) .* error(1:currLSize);
        % Iterate over every node in a layer
        for node = 1:nodeCounts(layer)
            weightCorrection = a * prevLValues * errorGrad(node);
            weightMatrix(node, 1:prevLSize, prevL, iteration + 1) = weightMatrix(node, 1:prevLSize, prevL, iteration) + weightCorrection;
        end
    end
    
    %weightTable(:, :, :, iteration + 1) = weightTable(:, :, :, iteration) + a * input(i,:) * error(iter);
    iteration = iteration + 1;
end

plot(errorSumSquared);