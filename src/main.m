clear
close all
clc
%rng(2077333712)

inputFile = dlmread("../data/features.txt");
desiredOutputFile = dlmread("../data/targets.txt");
values = zeros();
error = zeros();
errorSumSquared = zeros();

% Learning rate
a = 0.2;

nodeCounts = [10, 9, 11, 7];
layerCount = numel(nodeCounts);

% Generate a matrix with random weights for all the nodes in the network
weightMatrix = generateWeightMatrix(nodeCounts);

iteration = 1;

% Iterate over every entry in the input file
for inputNum = 1:size(inputFile,1)
    values(1,1:10) = inputFile(inputNum,:);
    
    % Iterate FORWARD over every layer in the network
    for layer = 2:layerCount
        % Iterate over every node in a layer
        for node = 1:nodeCounts(layer)
            % Truncate the inputs and outputs
            currInputs = values(layer - 1, 1:nodeCounts(layer - 1));
            currWeights = weightMatrix(node, 1:nodeCounts(layer - 1), layer - 1, iteration);
            values(layer, node) = perceptron(currInputs, currWeights, 1);
        end
    end
    
    desiredOutputs = full(vec2mat(ind2vec(desiredOutputFile(inputNum))',7));
    error(1:7) = desiredOutputs - values(1:7);
    errorSumSquared(inputNum) = sumsqr(error(1:7));
    
    % Iterate BACKWARD over every layer in the network
    for layer = layerCount:-1:2
        % Iterate over every node in a layer
        for node = 1:nodeCounts(layer)
            errorGrad = values(layer) * (1 - values(layer)) * error(1:7);
            disp(errorGrad)
            pause
        end
        error = oldError;
    end
    
    %weightTable(:, :, :, iteration + 1) = weightTable(:, :, :, iteration) + a * input(i,:) * error(iter);
end

plot(errorSumSquared);