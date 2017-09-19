clear
close all
clc
%rng(2077333712)

inputFile = dlmread("../data/features.txt");
desiredOutputs = dlmread("../data/targets.txt");
outputs = zeros();

nodeCounts = [10, 9, 11, 7];
layerCount = numel(nodeCounts);

% Generate a matrix with random weights for all the nodes in the network
weightMatrix = generateWeightMatrix(nodeCounts);

% Iterate over every entry in the input file
for inputNum = 1:size(inputFile,1)
    inputs = inputFile(inputNum,:);
    % Iterate over every layer in the network
    for layer = 2:layerCount
        % Iterate over every node in a layer
        for node = 1:nodeCounts(layer)
            % Truncate the inputs and outputs
            currInputs = inputs(1:nodeCounts(layer - 1));
            currWeights = weightMatrix(node, 1:nodeCounts(layer - 1), layer - 1);
            outputs(node) = perceptron(currInputs, currWeights);
        end
        inputs = outputs;
    end
    disp(outputs);
end