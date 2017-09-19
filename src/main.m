clear
close all
clc
%rng(2077333712)

inputFile = dlmread("../data/features.txt");
desiredOutputFile = dlmread("../data/targets.txt");
outputs = zeros();
error = zeros();

% Learning rate
a = 0.2;

nodeCounts = [10, 9, 11, 7];
layerCount = numel(nodeCounts);

% Generate a matrix with random weights for all the nodes in the network
weightMatrix = generateWeightMatrix(nodeCounts);

iteration = 1;

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
    desiredOutputs = full(vec2mat(ind2vec(desiredOutputFile(inputNum))',7));
    error(inputNum,1:7) = desiredOutputs - outputs(1:7);
    
    %weightTable(:, :, :, iteration + 1) = weightTable(:, :, :, iteration) + a * input(i,:) * error(iter);
    disp(outputs);
end