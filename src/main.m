clear
close all
clc
%rng(2077333712)

%read files 
inputFile = dlmread("../data/features.txt");
desiredOutputs = dlmread("../data/targets.txt");
outputs = zeros();

%Retrieve size of the matrix 
[m, n] = size(inputFile);
%disp(m);


threeFifth = 3/5*m;
fourFifth = 4/5*m;

%Define training set for features file (3/5th of total set)
trainingFeature = inputFile(1:threeFifth, :);

%Define validation set for feature file (1/5th of total set)
validationFeature = inputFile(threeFifth:fourFifth, :);

%Define test set for features file (1/5h of total set)
testFeature = inputFile(fourFifth:m, :);

%Define training set for targets file (3/5th of total set)
trainingTarget = desiredOutputs(1:threeFifth, :);

%Define validation set for targets file (1/5th of total set)
validateTarget = desiredOutputs(threeFifth:fourFifth, :);

%Define test set for targets file (1/5th of total set)
testTarget = desiredOutputs(fourFifth:m, :);



nodeCounts = [10, 15, 7];
layerCount = numel(nodeCounts);



% Generate a matrix with random weights for all the nodes in the network
weightMatrix = generateWeightMatrix(nodeCounts);

[weight1, weight2, error] = learningFunc(trainingFeature, trainingTarget);

% Iterate over every entry in the input file
%for inputNum = 1:size(inputFile,1)
   % inputs = inputFile(inputNum,:);
    % Iterate over every layer in the network
   % for layer = 2:layerCount
        % Iterate over every node in a layer
      %  for node = 1:nodeCounts(layer)
            % Truncate the inputs and outputs
           % currInputs = inputs(1:nodeCounts(layer - 1));
            %currWeights = weightMatrix(node, 1:nodeCounts(layer - 1), layer - 1);
            %outputs(node) = perceptron(currInputs, currWeights);
        %end
       % inputs = outputs;
   % end1
    %disp(outputs);
%end

